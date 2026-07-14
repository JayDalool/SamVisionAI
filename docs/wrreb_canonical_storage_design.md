# WRREB Canonical Storage Design (SamVisionAI v2)

Status: design + reviewable implementation. Companion to
[`wrreb_report_data_contract.md`](wrreb_report_data_contract.md). Nothing in this
phase applies migrations, writes to the production database, or touches the model.

## 1. Where this sits

The v1 ingestion branch (merged in `1175d66` / PR #4) produces a **dry-run
pipeline output** from the two WRREB Matrix reports. It never touches a database.
This phase adds the durable side: a reviewable schema, a storage/repository
layer, and a **default-dry-run staging loader** that turns pipeline output into
reviewable staging records, with a gated path to approved canonical sales.

```
WRREB PDFs (licensed, never in git; live under
  /srv/server/private/samvisionai/wrreb/incoming/)
        │  samvision.ingestion.import_wrreb_batch  (dry-run, existing)
        ▼
pipeline output  (gitignored: accepted.csv, rejected.csv, conflicts.csv,
  warnings.csv, extraction_diagnostics.csv, summary.json)
        │  samvision.storage.load_wrreb_batch  (dry-run by default; NEW)
        ▼
import_batches  +  staging_sales  +  import_issues     (staging schema)
        │  human issue review (resolve conflicts / rejections)
        ▼
samvision.storage.promote_wrreb_batch  (explicitly authorized; NEW)
        ▼
canonical_sales    ── read-only ──▶  dataset access layer (comparables / future training)
```

Two independent authorization gates, each requiring an explicit DB URL **and** an
explicit flag:

- staging write: `--authorize-staging-write`
- canonical write: `--authorize-canonical-write`

There is **no** `--authorize-production` flag in this phase. Neither loader reads
`SAMVISION_DB_*` production env vars; the DB URL is always passed in explicitly.

## 2. Architecture decisions (Phase 1 findings)

- **No migration framework exists.** The legacy `data/load_to_postgresql.py` uses
  raw `psycopg2` with an inline `CREATE TABLE IF NOT EXISTS housing_data`.
  SQLAlchemy 2.0.41 is a declared dependency but is used only for `create_engine`
  + raw SQL reads (`ml/train_model.py`, `ml/benchmark_models.py`,
  `app/streamlit_app.py`). Introducing Alembic would add a heavyweight dependency
  and an import-time metadata/target wiring that this codebase does not otherwise
  use. **Decision: hand-written, reviewable, reversible SQL files under
  `db/migrations/`** (paired `NNN_*.up.sql` / `NNN_*.down.sql`). This matches the
  existing raw-SQL idiom and applies nothing automatically.
- **SQLAlchemy Core (not ORM, not raw driver).** Core gives a typed schema,
  bound parameters, and explicit transactions while mapping generic types
  (`Numeric`, `Date`, `BigInteger`, `String`, `Boolean`) onto **both** PostgreSQL
  (production) and **SQLite** (synthetic tests — no `psycopg2`, no live Postgres,
  and it sidesteps the Python 3.14 wheel gaps). The ORM's implicit sessions and
  identity map add no value here; raw `psycopg2` (the legacy style) cannot be
  exercised against SQLite in tests.
- **Reuse:** `samvision/ingestion/provenance.py` hashing, `normalize_mls` from
  `samvision/ingestion/models.py`, and the ingestion output contract
  (`accepted.csv` columns, `summary.json` shape) as the loader's input format.
- **Retire later (not now):** `data/load_to_postgresql.py` and the `housing_data`
  `UNIQUE(address, listing_date)` identity, once `canonical_sales` is the source
  of truth for the model. Out of scope for this branch.
- **housing_data vs canonical conflicts:** different identity
  (`address+listing_date` vs MLS/LINC), different column names
  (`neighborhood/house_type/built_year/dom_days/sqft` vs
  `neighbourhood/property_type/year_built/dom/living_area_sqft`), fabricated
  defaults (`0/'none'/1.0`) vs nullable optionals, `BIGINT` vs `NUMERIC` prices,
  and derived columns (`season`, `sell_list_ratio`, lat/long) absent from
  canonical. The canonical schema is therefore a **new, separate** set of tables;
  `housing_data` is left untouched.

## 3. Identity model

- **MLS number** is the unique transaction / listing identifier. It is the unique
  key of `staging_sales` (per batch) and of `canonical_sales` (globally).
- **LINC** is the preferred *property* identifier. Many sales (over time) can
  share one LINC — a different MLS with the same LINC is **not** a duplicate; it
  may be a relisting, a later resale, or a separate legitimate transaction. LINC
  is indexed but **not** unique.
- **normalized address + postal code** is the fallback property identifier when
  LINC is missing. Stored as `normalized_property_id` (= LINC if present, else the
  normalized address+postal key) and indexed.
- Identity is **never** `address + sold_price` or `address + listing_date`.

## 4. Fingerprints

- **`batch_fingerprint`** — deterministic, unique. `sha256` over a canonical JSON
  of `{single_line_sha256, client_full_sha256, parser_version, contract_version}`.
  Re-importing the same two files with the same parser + contract yields the same
  fingerprint and is rejected as a duplicate batch. `batch_uuid` is a random UUID
  for external reference (not identity).
- **`record_fingerprint`** — deterministic per sale. `sha256` over the identifying
  + critical value fields: `mls_number, linc_number, normalized_property_id,
  sold_date, sold_price, list_price, address, postal_code`. Drives promotion
  idempotency (same MLS + same fingerprint → no-op) and conflict detection (same
  MLS + different fingerprint on a critical value → block).

## 5. Tables

Statuses/severities are stored as `TEXT` with `CHECK` constraints (portable to
SQLite) rather than PostgreSQL `ENUM`. Audit timestamps default to `now()` — that
is not a fabricated *data* value. All optional **property** fields are nullable
with no default; prices/tax are `NUMERIC`; `sold_date` is a real `DATE`.

### 5.1 `import_batches`
One row per imported report pair.

| column | type | notes |
|---|---|---|
| id | integer PK | |
| batch_uuid | text unique not null | random UUID, external ref |
| batch_fingerprint | text **unique** not null | §4 |
| contract_version | text not null | e.g. `wrreb-canonical/1.0` |
| parser_version | text not null | e.g. `single_line/1.0.0+client_full/1.0.0` |
| single_line_filename | text not null | basename only (no path leak) |
| single_line_sha256 | text not null | |
| client_full_filename | text not null | basename only |
| client_full_sha256 | text not null | |
| search_start_date | date null | batch coverage window (from sold-date range) |
| search_end_date | date null | |
| imported_at | timestamp not null default now() | |
| imported_by | text null | operator label, not PII |
| status | text not null default `parsed` | CHECK ∈ {parsed, staged, review_required, approved, rejected, failed} |
| single_line_count | integer not null default 0 | |
| client_full_count | integer not null default 0 | |
| matched_count | integer not null default 0 | |
| accepted_count | integer not null default 0 | |
| rejected_count | integer not null default 0 | |
| warning_count | integer not null default 0 | |
| conflict_count | integer not null default 0 | |
| needs_ocr_count | integer not null default 0 | |
| approved_at | timestamp null | |
| approved_by | text null | |
| notes | text null | operational, non-PII |

### 5.2 `staging_sales`
Accepted candidate rows for one batch (count == `accepted_count`).

Key columns: `id PK`, `batch_id FK→import_batches(id)`, `mls_number`,
`linc_number`, `normalized_property_id`, address/postal/area/neighbourhood,
`list_price NUMERIC(12,2)`, `sold_price NUMERIC(12,2)`, `sold_date DATE`, `dom`,
`property_type`, `style`, `year_built`, `living_area_sqft`,
`bedrooms_above_grade`, `bedrooms_total`, `full_bathrooms`, `half_bathrooms`,
`basement_type`, `basement_development`, `finished_basement_sqft`,
`lot_front_ft NUMERIC(8,2)`, `lot_depth_ft NUMERIC(8,2)`,
`lot_area_sqft NUMERIC(12,2)`, `garage_type`, `parking_description`,
`parking_spaces`, `new_construction BOOLEAN`, `gross_tax NUMERIC(12,2)`,
`tax_year`, `source_single_line_page`, `source_client_full_page`,
`parser_version`, `validation_status` (CHECK ∈ {accepted, warning}),
`record_fingerprint`, `created_at`.

- **`UNIQUE(batch_id, mls_number)`** — no duplicate MLS inside a batch.
- Indexes: `sold_date`, `linc_number`, `normalized_property_id`, `record_fingerprint`.

### 5.3 `import_issues`
Every warning / conflict / rejection / needs-OCR finding for review.

`id PK`, `batch_id FK→import_batches(id)`,
`staging_sale_id FK→staging_sales(id) null` (null for rejected/batch-level),
`mls_number`, `severity` (CHECK ∈ {warning, conflict, rejection, needs_ocr}),
`reason_code`, `field_name`, `single_line_value`, `client_full_value`,
`message`, `source_page`, `resolved BOOLEAN default false`, `resolution_notes`,
`resolved_at`, `resolved_by`, `created_at`.

- Index: `(batch_id, severity, resolved)`.
- `single_line_value` / `client_full_value` hold only the **conflicting canonical
  field values** (dates, prices, addresses) — never remarks/contact/lockbox text.

### 5.4 `canonical_sales`
Approved sales. Promotion target; read source for comparables/training.

Same property/value columns as `staging_sales`, plus `source_batch_id
FK→import_batches(id)`, `record_fingerprint`, `data_quality_status` (CHECK ∈
{approved, flagged}), `created_at`, `updated_at`.

- **`UNIQUE(mls_number)`** — one canonical row per transaction/listing.
- Indexes: `sold_date`, `neighbourhood`, `area_code`, `linc_number`,
  `normalized_property_id`, `sold_price`, `property_type`.

## 6. Loader & promotion behaviour

**Staging loader** (`load_wrreb_batch`, default dry-run):
- Validates presence of all manifest files (`summary.json` + the five CSVs).
- Re-checks summary counts against actual CSV row counts (mismatch → refuse).
- Recomputes `batch_fingerprint`; refuses on duplicate.
- Detects duplicate MLS within the batch.
- Refuses batches with `critical_reconciliation_failed = true`, unresolved
  conflicts, or an unrecognized `contract_version`.
- A DB write requires **both** a non-empty explicit `--database-url` and
  `--authorize-staging-write`; a URL silently inferred from production env is
  refused. The write runs in **one transaction**; any failure rolls back
  everything and exits non-zero with a privacy-safe reason.

**Promotion** (`promote_wrreb_batch`, explicitly authorized) — see §7 rules.
Implemented as a small, transactional repository operation; **not executed**
against any database in this phase.

## 7. Promotion rules (idempotency & identity)

A batch may promote only when: status is `staged`, no unresolved `rejection`, no
unresolved `conflict`, critical reconciliation passed, and
`accepted_count == count(staging_sales)`, with an explicit authorization flag.

Per record:
- same MLS + same `record_fingerprint` → **idempotent no-op**;
- same MLS + different critical value → **block + create a `conflict` issue**
  (never overwrite silently);
- different MLS + same LINC → **preserved as another transaction**;
- missing LINC → property identity falls back to normalized address+postal.

## 8. Privacy

Never stored in any table, logged, or committed in fixtures: agent names,
brokerage names, phone numbers, emails, showing/appointment instructions, lockbox
info, private operational notes, full remarks. The parsers already exclude these;
storage model objects reject them defensively (a guarded constructor / allowed-key
check), and tests assert they cannot enter canonical objects. Filenames are stored
as **basenames only** so absolute private paths never leak.

## 9. Rollback / reversibility

Each migration ships a `*.down.sql` that drops exactly what its `*.up.sql`
created, in reverse dependency order. Migrations are additive (only new tables)
and contain no destructive statements against existing objects. Applying and
rolling back is a documented manual step for a reviewer/DBA — never automatic and
never at application startup.
