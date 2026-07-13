# SamVision AI — Training Data Lineage Audit

_Audited: 2026-07-12. Scope: read-only. No production data, credentials, or the
production model artifact were modified. Companion to
[`current_model_audit.md`](current_model_audit.md) and
[`model_v2_design.md`](model_v2_design.md)._

## TL;DR

The data that trains the price model is **PDF-scraped MLS "Sold" reports**, not
the synthetic generator the model audit assumed. Tracing it end to end surfaces
four lineage defects that matter more than any modelling choice:

1. **`listing_date` is fabricated.** Both parsers stamp `datetime.today()` and
   throw away the real sale date that is present in the source PDFs. The live DB
   holds exactly **two distinct dates — both load-run days** (2026-02-28 and
   2026-05-26). Every time-based feature and the "chronological" benchmark split
   are therefore built on a load-batch label, not real time.
2. **The production artifact is unreproducible.** `model_explanation.json` claims
   `trained_on_rows = 20548`; no file or DB state in this repo contains anywhere
   near that many rows (DB: 4,364; largest CSV: 3,838). It was trained on a
   dataset that no longer exists.
3. **"Validation" is a no-op.** `clean_and_validate_csv.py` performs zero
   cleaning; the only "validation" is the DB loader silently filling missing
   fields with zeros/`"none"`, which hides bad rows instead of dropping them.
4. **The model audit's provenance note is wrong.** It attributes the DB rows to
   `data/generate_real_estate_data.py`. That generator's dates (2022–2023) and
   schema don't match the DB; it is an **orphaned, unused path**. Correct source
   is the `pdf_uploads → parsed_csv → PostgreSQL` chain below.

---

## 1. The lineage chain (as actually wired)

Row counts are the real record counts (some CSVs embed newlines inside address
fields, so `wc -l` over-counts wildly — the root CSV is 984 records, not 19,863).

```
pdf_uploads/*.pdf , *.txt                 raw MLS "Sold" report exports
        │
        │  scripts/parse_and_export.py  →  utils.pdf_sales_parser.extract_pdf_sales
        ▼
winnipeg_housing_data.csv  (repo root)     984 records, raw multi-line text
parsed_csv/from_txt_parser.csv             1,044 rows  (txt-parser variant)
        │
        │  data/merge_missing_data.py  (+ data/missing_addresses.txt)
        ▼
parsed_csv/merged.csv                      3,838 rows
        │
        │  data/clean_and_validate_csv.py   (NO-OP — pandas re-encode only)
        ▼
parsed_csv/validated.csv                   3,838 rows  (byte content == merged)
        │
        │  data/load_to_postgresql.py       ON CONFLICT (address, listing_date) DO NOTHING
        ▼                                    ensure_all_columns() fills defaults
PostgreSQL  housing_data                    4,364 rows  (2 load batches)
        │
        │  ml/train_model.py  (SELECT … FROM housing_data)
        ▼
trained_price_model.pkl
```

### Row count at each hop

| Stage | File / store | Records | Notes |
|---|---|---:|---|
| Parse (PDF) | `winnipeg_housing_data.csv` (root) | 984 | Addresses are raw multi-line PDF blocks; unusable as-is. |
| Parse (txt) | `parsed_csv/from_txt_parser.csv` | 1,044 | All dated `2025-07-02` (parse-run day). |
| Merge | `parsed_csv/merged.csv` | 3,838 | All dated `2026-02-28`. 165 neighborhoods. |
| "Validate" | `parsed_csv/validated.csv` | 3,838 | Content identical to `merged.csv` (see §3). |
| Load | DB `housing_data` | 4,364 | 3,615 on 2026-02-28 + 749 on 2026-05-26. |
| Train | `trained_price_model.pkl` | **20,548 (claimed)** | Not reproducible from any source above (§4). |

### Live DB snapshot (read-only query, 2026-07-12)

```
total rows: 4364
listing_date distribution:
  2026-02-28   3615
  2026-05-26    749
distinct address: 4351   distinct mls_number: 4351
```

Two dates, both load-run days; 13 duplicate addresses.

---

## 2. Defect #1 — `listing_date` is a load stamp, not a sale date

The single most consequential lineage problem. Real sale dates exist in the
source PDFs (each record has a `Sell Date: MM/DD/YYYY` field) but are discarded:

- `utils/pdf_sales_parser.py:317,345,359` — `today = datetime.today().date()`
  then `listing_date = today`. The parser stamps the run date on every row.
- `data/merge_missing_data.py:26` — regex **captures** `Sell Date:` into `date`
  … and then `:36` overwrites `listing_date` with `datetime.today()` and never
  uses `date`. The ground truth is parsed and thrown away one line later.

Because of this:

- The DB has only **two** `listing_date` values, each equal to a batch-load day.
- `ml/benchmark_models.py` "splits chronologically by `listing_date`" (`:14`,
  `:162`, `:443`). In reality this becomes **train = the 2026-02-28 load batch,
  test = the 2026-05-26 load batch** — a batch split, not a temporal one. It
  still avoids target leakage (its main purpose) but its "chronological
  generalization" framing does not hold.
- Every time-derived feature is meaningless on this data: `listing_month`,
  `season`, `season_boost`, and `recency_weight` all derive from a fabricated
  date. `season` is separately hard-coded to `"Spring"`/`"Summer"` in the
  parsers.

**Fix:** extract and persist the real `Sell Date` (already present in source
text) as the sale date; keep the load timestamp as a separate `loaded_at`
column. This is a prerequisite for any honest temporal split in v2.

---

## 3. Defect #2 — "validation" cleans nothing; the loader masks bad data

- `data/clean_and_validate_csv.py` reads `merged.csv`, writes `validated.csv`,
  and every cleaning step in between is commented out (`:19-21`). The "validated"
  artifact is a pandas re-encode of the merged file — same 3,838 rows, same
  values. The name overstates what happens.
- The only place data is coerced is `data/load_to_postgresql.py:54-89`
  (`ensure_all_columns`), which **fills missing fields with defaults**
  (`dom_days=0`, `built_year=0`, `list_price=0`, `sold_price=0`,
  neighborhood/address `"none"`) rather than rejecting incomplete rows. Bad or
  unparsed rows enter the DB as zero-valued records and become training noise.
  `sold_price=0`/`list_price=0` rows are only dropped later, in the benchmark,
  not before training.
- Parser output quality is low and unaudited: sample addresses are concatenated
  multi-line PDF text (`"34 Arnold Avenue 405 Churchill Drive 211 Morley
  Avenue"`, `"25 SF 50 SF 25 SF Depth 100 SF …"`), and `mls_number` values are
  random hex hashes (`e2b6c33ed3`), not real MLS numbers — so
  `UNIQUE(address, listing_date)` and any address-based join are unreliable.

---

## 4. Defect #3 — the production artifact can't be reproduced

`model_explanation.json` records `trained_on_rows = 20548`. The largest data
source in the repo is the 3,838-row `validated.csv`; the DB holds 4,364. There
is no path from today's data to a 20,548-row training set. The shipped
`trained_price_model.pkl` was trained on data that is **no longer present** —
lineage is severed at the artifact itself. Combined with the target leakage
documented in `current_model_audit.md`, the "Model Information" metrics describe
a model/dataset pair that cannot be regenerated or verified.

The ~70 timestamped `trained_price_model_*.pkl` snapshots in the repo root are
likewise unlabelled as to which data batch produced them.

---

## 5. Defect #4 — orphaned and conflicting data paths

- **Synthetic generator is unused.** `data/generate_real_estate_data.py` emits
  2022–2023 dates and a different schema (`age`, `garage` bool, no
  `address`/`mls_number`). It does not feed the DB and does not match DB
  contents. `current_model_audit.md:49` attributing the DB to this generator is
  incorrect and should be revised.
- **Two files named `winnipeg_housing_data.csv`.** The root copy (984 raw parser
  records) is written by `scripts/parse_and_export.py` (`OUT_CSV`), while
  `data/merge_missing_data.py` reads/writes a *different* `data/winnipeg_housing_data.csv`
  (2 records). The scripts don't share a path, so the "merge" step is not
  actually wired to the "parse" step — the pipeline is stitched together by hand.
- **No lineage metadata.** No `source_file`, no `loaded_at`, no batch id, no data
  version anywhere. `ON CONFLICT … DO NOTHING` means re-loads merge silently with
  unknown provenance, so the DB is a union of undated load events.

---

## 6. Recommendations

1. **Persist the real sale date.** Extract `Sell Date` in both parsers; stop
   stamping `datetime.today()`. Add a separate `loaded_at` column. Without this,
   no temporal split (v2 included) is trustworthy.
2. **Make validation real or delete it.** Replace the no-op
   `clean_and_validate_csv.py` with actual checks (drop rows with
   non-positive price/sqft, bad addresses, unparsed fields) *before* the DB load,
   and stop defaulting missing numerics to `0`.
3. **Record lineage.** Add `source_file` / `batch_id` / `loaded_at` columns so
   every DB row traces to the PDF and load event it came from.
4. **Version training sets.** Snapshot the exact rows used to train each artifact
   (row hash + count) into `model_explanation.json`; treat a model whose training
   data can't be reproduced as unshippable.
5. **Retire dead paths.** Remove or clearly mark
   `data/generate_real_estate_data.py` and the duplicate
   `winnipeg_housing_data.csv` files; wire parse → merge → validate → load
   through one explicit, configurable path.
6. **Correct the model audit** provenance note (`current_model_audit.md:47-49`)
   to point at the PDF pipeline, not the synthetic generator.

---

## Reproduce this audit

```bash
# Row counts / date spans of the CSV stages (stdlib only):
python3 - <<'PY'
import csv
for f in ["parsed_csv/from_txt_parser.csv","parsed_csv/merged.csv",
          "parsed_csv/validated.csv","winnipeg_housing_data.csv"]:
    rows=list(csv.DictReader(open(f,newline='',encoding='utf-8',errors='replace')))
    dates=sorted({r["listing_date"] for r in rows if r.get("listing_date")})
    print(f, len(rows), "dates:", dates[:1], "->", dates[-1:])
PY

# Live DB date distribution (read-only), via the app image on the docker network:
docker run --rm --network samvision-net --env-file .env \
  -v "$PWD":/work -w /work samvisionai-samvision-app python -c "
import psycopg2; from utils.db_config import get_db_config
c=get_db_config(); conn=psycopg2.connect(dbname=c['dbname'],user=c['user'],
  password=c['password'],host=c['host'],port=c['port']); cur=conn.cursor()
cur.execute('SELECT listing_date, COUNT(*) FROM housing_data GROUP BY 1 ORDER BY 1')
print(cur.fetchall())"
```
