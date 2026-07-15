# Comparable-Sales Leave-One-Out Backtest Design (SamVisionAI)

Status: design + implementation. A **read-only** leave-one-out (LOO) backtest of
the deterministic comparable-sales engine
([`comparable_sales_engine_design.md`](comparable_sales_engine_design.md)). It
measures how well the transparent engine reproduces **held-out** approved
canonical sold prices. It **never** trains, tunes, or changes scoring weights,
and it performs **no database writes**.

## 1. Purpose

Answer, on the validated canonical batch, questions a realtor pilot needs:
coverage (what fraction of sales get ≥ `MIN_USABLE_COMPS` comparables), accuracy
(MAE / median APE / within-threshold), and where the engine is weak
(neighbourhood, property type, tier, similarity, confidence). Output is
decision-support evidence, **not** an appraisal benchmark and **not** a signal to
auto-tune weights.

## 2. Method (per approved canonical sale)

For each approved canonical row treated as the **held-out subject**:

1. Treat the sale as the subject property.
2. `valuation_date = held-out sold_date` (never `today()`, never import date).
3. Exclude the subject MLS from candidates.
4. Consider only candidates with `sold_date <= valuation_date`.
5. Run the **unmodified production** `ComparableService.find_comparables`.
6. Compare `indicated_value` against the held-out **actual** `sold_price`.
7. Record metrics, tier, confidence, similarity, spread, warnings.
8. Never use a future sale (`sold_date > valuation_date` is rejected by engine
   eligibility and by the SQL upper bound).
9. Never train or tune on the held-out outcome — the actual price is used **only**
   to compute error after the estimate is produced.

The engine already enforces leakage safety: `service._search_tier` sets
`sold_end = valuation_date` and `exclude_mls = subject.mls_number`;
`storage.dataset.query_canonical_sales` filters `sold_date <= sold_end` and
`mls_number != exclude_mls`; `scoring.eligibility` rejects `future_sale` and
`subject_self`. The backtest adds no new chronology of its own.

### Same-date chronology limitation (important)

The upper bound is **inclusive** (`sold_date <= valuation_date`). A different
sale closing on the **same calendar date** as the held-out subject is therefore
eligible as a comparable. The canonical dataset has **no intraday timestamp**, so
same-day ordering cannot be established and such a comp cannot be excluded without
also discarding legitimately-prior same-day evidence. This is a **known, bounded
optimism** in the backtest and is reported, not silently hidden. (The subject
itself is always excluded by MLS.)

## 3. Subject eligibility

A held-out row is **eligible** only when it has: valid `mls_number`; `approved`
status; `sold_date`; `sold_price > 0`; `living_area_sqft > 0`; `property_type`;
and `area_code` **or** `neighbourhood`. Optional subject fields may be missing.

Ineligible rows are **skipped** and counted with a reason code (never valued):

| Reason code | Condition |
|---|---|
| `MISSING_MLS` | no MLS number |
| `MISSING_SOLD_DATE` | no sold_date |
| `INVALID_SOLD_PRICE` | sold_price missing or ≤ 0 |
| `INVALID_LIVING_AREA` | living_area_sqft missing or ≤ 0 |
| `MISSING_PROPERTY_TYPE` | no property_type |
| `MISSING_GEOGRAPHY` | neither area_code nor neighbourhood |
| `INSUFFICIENT_COMPARABLES` | eligible, but engine returned `insufficient_data` (< `MIN_USABLE_COMPS`) — counts toward coverage denominator, not accuracy |
| `ENGINE_ERROR` | engine raised; captured, never fatal |

`INSUFFICIENT_COMPARABLES` and `ENGINE_ERROR` occur **after** the subject is
found eligible; the first six are pre-run eligibility skips.

## 4. Metrics

Money is `Decimal`; deterministic rounding (`ROUND_HALF_UP`). Reject
`actual_sold_price <= 0` before any ratio.

- Per case: absolute error `|indicated − actual|`; percentage error
  `|indicated − actual| / actual`; squared error.
- Aggregate over **valued** cases: MAE, median absolute error, RMSE, MAPE,
  median absolute percentage error.
- Threshold accuracy: share within 5 % / 10 % / 15 % / 20 % APE.
- Distribution of absolute error: min, p25, median, p75, p90, p95, max.
- Coverage rate = valued cases / eligible held-out subjects.

## 5. Segmented evaluation

Grouped metrics by: neighbourhood, area_code, property_type, confidence level,
geographic tier, selected-comparable-count bucket (`3`, `4–5`, `6–7`, `8–10`),
median-similarity bucket (`<50`, `50–59.99`, `60–69.99`, `70–79.99`, `80+`),
data-coverage bucket, and sold month. Groups below the minimum sample size
(`3`) are still shown but explicitly marked **unstable**.

## 6. Failure-case analysis

Largest-error cases are reported **sanitized**: anonymous index, MLS masked to the
last 3 digits, area_code, neighbourhood, property_type, sold_date, actual price,
indicated value, absolute + percentage error, confidence, tier, comparable count,
median similarity, warning codes, missing subject fields, and a heuristic failure
category (`INSUFFICIENT_COMPS`, `WIDE_PRICE_SPREAD`, `GEOGRAPHIC_WIDENING`,
`LOW_SIMILARITY`, `LOW_DATA_COVERAGE`, `PROPERTY_OUTLIER`, `MARKET_TIMING`,
`SIZE_MISMATCH`, `UNKNOWN`).

## 7. Privacy

Reports **never** contain: full street address, `linc_number`, `postal_code`,
`normalized_property_id`, or any `PRIVACY_BANNED_FIELDS` (agent/brokerage/phone/
email/remarks/showing/lockbox). MLS is **masked** (last 3 digits) in every written
report. Output is written only to an explicit, non-tracked directory (default
`/tmp/...`); the CLI refuses to write into tracked source directories, and no
report artifact is committed.

## 8. Package layout

```
samvision/backtesting/
  __init__.py   models.py   metrics.py   runner.py   reporting.py
  run_comparable_backtest.py   (read-only CLI)
```

`runner.py` performs no writes and holds no write-authorization flag; it depends
only on a `ComparableService` (any `ComparableDataSource`). Tests use synthetic
in-memory data — no database required.

## 9. Limitations

100 sales over ~3 months is a **small, single-batch** sample: many groups are
below the stability threshold, early-window subjects have few or zero prior
comps, and `garage`/`lot`/`parking`/`basement_development` are absent so those
scoring dimensions never contribute. Results are **directional evidence for a
pilot decision**, not a production accuracy guarantee, and must not be used to
auto-tune weights.
