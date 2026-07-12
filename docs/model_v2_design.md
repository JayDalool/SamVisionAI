# SamVision AI — Valuation Model v2 Design

_Status: proposal. No production model is replaced by this document. Promotion is
gated on the criteria in [§7](#7-promotion-gate)._

This design responds to the leakage and train/serve-mismatch findings in
[`current_model_audit.md`](./current_model_audit.md). The goal is a valuation
system a realtor can trust: honest accuracy, an explicit uncertainty range,
comparable evidence, and a **clear separation between what a home is worth and
what to offer for it**.

---

## 1. Three separate outputs (do not conflate them)

The current app produces one number ("Recommended Winning Offer") that silently
mixes a price estimate with an offer-strategy premium. v2 splits this into three
distinct, independently-validated outputs:

| # | Output | Question it answers | Basis |
|---|---|---|---|
| 1 | **Estimated market value** | "What is this property worth?" | Valuation model on pre-sale features. Independent of any list price strategy. |
| 2 | **Expected sale-price range** | "What will it likely sell for?" | Value + calibrated prediction interval (e.g. P10–P90). |
| 3 | **Offer-strategy range** | "What should I offer to win?" | A *separate* layer using list price + historical market behaviour (multi-offer premiums, DOM patterns). Never presented as guaranteed. |

The UI must label these separately and must **not** imply a guaranteed winning
offer.

Alongside the numbers, always show:
- **Confidence range** (the interval, with its validated coverage).
- **Comparable properties** that drove the estimate, and *why* each was chosen.
- **Important factors** (feature contributions) for this prediction.
- **Out-of-coverage warnings** when inputs fall outside the training distribution
  (unseen neighborhood, sqft/price far outside observed range, etc.).

---

## 2. Feature schema (leakage-free)

**Allowed (known before sale):**

- Numeric: `bedrooms`, `bathrooms`, `sqft`, `lot_size`, `built_year`, `age`
  (from `listing_date.year − built_year`), `list_price`, `listing_month`.
- Categorical: `neighborhood`, `house_type`, `style`, `garage_type`,
  `basement_type`, `season`.
- Optional derived (must be fit on **train folds only**, inside the pipeline):
  neighborhood price level, comp density.

**Forbidden (leak the target or are post-sale):**

`sold_price`, `price_per_sqft` (from sold), `price_diff`, `over_asking_pct`,
`sell_list_ratio`, `dom_days`, and anything computed from full-dataset statistics
before splitting. `list_price` is allowed for valuation but its influence should
be monitored (see §6) and it is the primary input for the offer layer, not the
value estimate.

---

## 3. Architecture

### A. Comparable-sales engine
Retrieve recent, similar sold properties and score similarity:

- Candidate filter: same/nearby `neighborhood`, same `house_type`, recent
  `listing_date`.
- Similarity score over `neighborhood`, `house_type`, `sqft`, `bedrooms`,
  `bathrooms`, `age`, `garage_type`, and **recency** (recency decay so newer
  sales weigh more).
- Output the top-N comps with **per-comp reasons** ("same neighborhood, ±150
  sqft, sold 3 weeks ago") and a comparable-implied value (recency-weighted
  median \$/sqft × subject sqft). This doubles as an explainable baseline
  (benchmarked as `baseline_comparable_ppsf`, MAE ≈ \$115k today — the models
  must clearly beat it, and they do).

### B. Valuation model
- Single `sklearn` **Pipeline** = `ColumnTransformer` (impute + one-hot
  categoricals; impute + scale numerics) **+** estimator, so **preprocessing is
  identical for training and inference** (the current train/serve mismatch is
  structurally impossible).
- Benchmark set (`ml/benchmark_models.py`), all leakage-free, chronological:
  1. simple median / comparable \$/sqft baseline,
  2. regularized linear regression (Ridge),
  3. RandomForest / ExtraTrees,
  4. HistGradientBoosting,
  5. LightGBM/XGBoost only if already installed and justified (LightGBM is).

  **Current benchmark leader:** Ridge (MAE \$32,726, R² 0.961) ≈ RandomForest
  (\$32,729). Both already beat the honest production model (\$38,248).

### C. Uncertainty
- Produce a **range**, not just a point estimate.
- Preferred: **conformal prediction** (split-conformal on the validation fold)
  for distribution-free intervals; or **quantile models** (e.g. HistGradientBoosting
  with `loss="quantile"` at α=0.1/0.5/0.9, or LightGBM quantile objective).
- **Validate interval coverage**: a nominal 80% interval must contain the true
  `sold_price` ~80% of the time on the chronological test set (report empirical
  coverage and mean interval width). An interval that isn't calibrated is not
  shipped.

### D. Offer strategy (kept separate)
- Inputs: `list_price`, neighborhood multi-offer premium priors, and historical
  market behaviour (e.g. observed sale/list ratios, DOM distributions) — used at
  **strategy** time, not baked into the value model.
- Output: an offer *range* with rationale, explicitly framed as guidance.
- **Never** claim a guaranteed winning offer (current UI disclaimer must remain
  and be strengthened).

---

## 4. Evaluation protocol

- **Chronological split** by `listing_date` (train → val → test).
- **Duplicate-property separation**: an address lives in exactly one split.
- Report, overall and per segment (neighborhood, price band, house type, sqft
  band): **MAE, median absolute error, RMSE, MAPE (where valid), R²**.
- Report **interval coverage** and **mean interval width** for the uncertainty
  layer.
- Fixed random seed; reproducible via `python -m ml.benchmark_models`.

---

## 5. Reproducible benchmark command

```bash
python -m ml.benchmark_models        # writes reports/model_benchmark_<ts>.{json,csv}
                                      # and reports/model_segment_errors_<ts>.csv
```

The command already: loads via the central DB config, exposes no secrets,
validates required columns, splits chronologically, prevents duplicate-property
leakage, uses a fixed seed, writes timestamped reports, and **never overwrites
the production model**.

---

## 6. Guardrails / monitoring

- **Out-of-coverage warnings** when an input is outside training support
  (unseen category level, `sqft`/`list_price`/`age` outside observed quantile
  range) → widen or suppress the estimate.
- **List-price dependence check**: track how much the value estimate moves with
  `list_price` alone; a valuation that merely echoes list price is flagged.
- **Sanity vs. comparables**: if the model estimate diverges sharply from the
  comparable-implied value, surface the disagreement rather than hiding it.

---

## 7. Promotion gate

A candidate model replaces `trained_price_model.pkl` **only if all hold**:

1. It **beats the current model on chronological MAE _and_ median absolute
   error** on the held-out test set.
2. **Schema/leakage tests pass** (no forbidden feature present; identical
   train/inference preprocessing).
3. The **saved artifact loads** cleanly via `joblib.load`.
4. **Inference works from the real UI** end-to-end (drive the prediction form,
   get a sane number).
5. **Prediction intervals are validated** (empirical coverage ≈ nominal).

Promotion is a **separate commit** from this design, from the audit, and from the
form-state fix. Until the gate is met, the production model is left untouched.
