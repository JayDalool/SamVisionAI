# Comparable-Sales Engine Design (SamVisionAI)

Status: design + implementation. A **transparent, deterministic** comparable-sales
baseline built **only** on approved `canonical_sales`. It is **decision support,
not an appraisal**, and it is **not** a trained ML model. Companion to
[`wrreb_canonical_storage_design.md`](wrreb_canonical_storage_design.md).

## 1. Flow

```
Subject property (typed, validated)
  → eligibility filters   (approved canonical only, sold ≤ valuation_date, ...)
  → geographic search tiers (neighbourhood → area, 365d → 730d; narrowest first)
  → deterministic similarity scoring (0–100, fixed weights, no learning)
  → stable ranked comparables
  → transparent price indicators (weighted medians, size-normalized)
  → rule-based confidence + warnings
```

## 2. Architecture review (Phase 1 findings)

- **Reliable canonical fields** (populated for the validated 100-sale batch):
  `mls_number, area_code, neighbourhood, property_type, style, year_built,
  living_area_sqft, bedrooms_above_grade, bedrooms_total, full_bathrooms,
  half_bathrooms, basement_type, sold_price, sold_date, list_price, dom,
  gross_tax, tax_year, linc_number, postal_code, address`.
- **Mostly / entirely NULL today** (parser does not yet populate): `garage_type,
  parking_spaces, parking_description, new_construction, basement_development,
  finished_basement_sqft, lot_area_sqft, lot_front_ft, lot_depth_ft`. These are
  **optional** scoring dimensions — a missing value scores **0** for that
  dimension (never treated as a match) and lowers the record's data-coverage.
- **Legacy valuation code that must NOT be reused**: `ml/train_model.py`,
  `ml/benchmark_models.py`, `trained_price_model*.pkl`,
  `utils/compute_expected_multi_offer_premium.py`,
  `utils/prediction_form.py`. Those are an opaque premium/ML path; this engine is
  independent, deterministic, and explainable. **No trained model is used for
  scoring or valuation.**
- **Data source**: the read-only `samvision.storage.dataset` layer
  (`query_canonical_sales`, `CanonicalDataset`), extended with an `exclude_mls`
  filter. The engine depends on a small dataset **protocol** with two
  implementations (Postgres-backed and in-memory) so unit tests need no database.

### MVP required subject fields
`valuation_date`, (`area_code` **or** `neighbourhood`), `property_type`,
`living_area_sqft`. All others optional.

### Leakage / future-sale prevention
Chronology uses **`sold_date` only**. A candidate is eligible only if
`sold_date <= valuation_date`. `listing_date`, import date, model-training date,
and **today()** are never used as transaction chronology. `valuation_date` is
**never** defaulted from today — the caller must supply it. Ages are computed at
historical dates (subject age at `valuation_date.year`, comparable age at
`sold_date.year`), never at the current calendar year.

### Explainability
Every comparable carries its component scores, data-coverage, geographic tier,
sold-date age, key differences from the subject, human-readable reasons, and
warnings. The valuation is a weighted median of transparent, size-normalized
sold prices — a realtor can reproduce it by hand.

## 3. Subject-property contract

Typed `SubjectProperty` (see `models.py`) with: `valuation_date`, optional
`mls_number`/`linc_number`/`address`/`postal_code`, `area_code`, optional
`neighbourhood`, `property_type`, optional `style`/`year_built`,
`living_area_sqft`, optional `bedrooms_total`/`full_bathrooms`/`half_bathrooms`/
`basement_type`/`basement_development`/`lot_area_sqft`/`garage_type`/
`parking_spaces`.

Validation (raises `SubjectValidationError`): `valuation_date` is a real date;
`living_area_sqft > 0`; `year_built` plausible (1800..valuation_date.year) and
not after the valuation year; bedroom/bathroom counts ≥ 0; at least one of
`area_code`/`neighbourhood` present; `property_type` present.

## 4. Eligibility rules

A candidate is eligible only if it: comes from **approved** `canonical_sales`;
has `sold_date <= valuation_date`; `sold_price > 0`; a valid MLS number; matches
the required `property_type`; is within the configured max lookback; is not the
subject MLS (when supplied); and has `living_area_sqft > 0` (size comparison is
required for the MVP). Future sales, staging rows, rejected/unapproved records,
`listing_date`/import/model/current-date chronology are all excluded.

## 5. Geographic search tiers (`constants.py`)

Narrowest first; widen only if `MIN_USABLE_COMPS` not met; never silently beyond
area code; the selected tier is reported and widening raises a warning.

| Tier | Geography | Property type | Lookback |
|------|-----------|---------------|----------|
| 1 | exact neighbourhood | exact | 365 days |
| 2 | exact area code | exact | 365 days |
| 3 | exact area code | exact | 730 days |

Candidates are deduplicated by MLS across tiers. Defaults: `MIN_USABLE_COMPS=3`,
`PREFERRED_COMPS=5`, `MAX_RETURNED_COMPS=10`, `MAX_LOOKBACK_DAYS=730`. (If the
subject has no neighbourhood, Tier 1 is skipped and Tier 2 is the entry tier.)

## 6. Deterministic similarity scoring (0–100)

Total = Σ (weight × component_subscore), each subscore in [0,1]. A **missing**
optional dimension contributes **0** (not skipped, not a match) so sparse records
cannot outrank complete ones. Final weights (documented, sum = 100):

| Dimension | Weight | Subscore rule |
|-----------|-------:|---------------|
| geography | 25 | exact neighbourhood → 1.0; same area only → 0.5; else ineligible |
| living area | 25 | `max(0, 1 − pctdiff/0.50)`; exact → 1, ≥50% diff → 0 |
| recency | 20 | `max(0, 1 − age_days/730)`; newest → 1, ≥730d → 0 |
| bedrooms | 8 | `max(0, 1 − |Δ|/3)`; both present else 0 |
| bathrooms | 8 | baths = full + 0.5·half; `max(0, 1 − |Δ|/3)`; both present else 0 |
| year-built/age | 6 | `max(0, 1 − |Δage|/50)`; ages at valuation/sold years; both present else 0 |
| style | 4 | exact (both present) → 1 else 0 |
| garage | 2 | exact (both present) → 1 else 0 |
| basement | 2 | `basement_type` exact (both present) → 1 else 0 |

`data_coverage` = fraction of optional dimensions present on the candidate
(bedrooms, bathrooms, year_built, style, garage, basement), reported 0–1.

Stable tie-break: (1) similarity desc, (2) `sold_date` desc, (3) `mls_number`
asc.

## 7. Price indicators (`statistics.py`, Decimal-safe)

For each comparable: `sold_price`, `ppsf = sold_price / living_area_sqft`,
`size_normalized_price = sold_price × subject_sqft / comp_sqft`. Weights =
similarity scores. Using Decimal-safe **weighted median** and **weighted
quantiles** (25th/75th):

- `indicated_value` = weighted median of `size_normalized_price` (primary
  baseline when subject sqft present), rounded to the nearest **$500** (no false
  precision).
- `indicated_range_low/high` = weighted 25th/75th percentiles of
  `size_normalized_price`.
- `direct_weighted_median_price` = weighted median of raw `sold_price`.
- `weighted_median_ppsf` = weighted median of `ppsf`.
- `price_spread` = `(p75 − p25) / indicated_value`.
- `median_similarity` = plain median of the selected similarity scores.

Guards: reject zero/None sqft and non-positive prices before ratios; ratios use
Decimal; extreme ratios are excluded from indicators via a documented sanity band
(0.2×–5× subject ppsf) and raise a warning.

## 8. Confidence (rule-based, not a probability)

Implemented thresholds:

- **High**: ≥ 7 comps **and** median similarity ≥ 75 **and** mean coverage ≥ 0.75
  **and** price spread ≤ 0.20 **and** tier 1 (no widening beyond neighbourhood).
- **Medium**: ≥ 4 comps **and** median similarity ≥ 60 **and** mean coverage ≥
  0.50 **and** price spread ≤ 0.35.
- **Low**: anything else that still has ≥ `MIN_USABLE_COMPS` comps.
- **Insufficient**: fewer than `MIN_USABLE_COMPS` (3) usable comps → **no point
  estimate**; returns the available records with an `insufficient_data` warning.

## 9. Package layout

```
samvision/comparables/
  __init__.py     constants.py   models.py     datasource.py
  scoring.py      statistics.py  service.py    explain.py   find_comps.py
```

The service depends on the `ComparableDataSource` protocol (Postgres or
in-memory). It performs no writes; there is **no** authorization-to-write flag —
the CLI can only read.

## 10. Limitations (also surfaced by the CLI)

This is a **comparable-sales decision-support tool, not an appraisal**. It uses
only historical **approved** canonical sales and **excludes future sales**. It
does **not** adjust for renovations, interior condition, exact micro-location,
school catchment, lot quality, view, or market trend; it does not use current
listings or active competition. Missing subject/comparable fields reduce
confidence; widening geography or time reduces confidence. **Results require
realtor review.**
