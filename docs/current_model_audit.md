# SamVision AI — Current Prediction Model Audit

_Audited: 2026-07-12. Scope: read-only. No production data, credentials, or the
production model artifact were modified._

## TL;DR

The production "winning offer price" model reports **MAE ≈ \$866** and
**R² ≈ 0.99997** (`model_explanation.json`). Those numbers are **not real** — they
are the product of **target leakage**. Three of the model's features are computed
directly from the target (`sold_price`), so at training time the model can
essentially read the answer.

When the exact same production artifact (`trained_price_model.pkl`) is evaluated
**honestly** — on a chronological hold-out, fed the same placeholder inputs the
live UI supplies — its error is:

| Metric | Reported (leaky) | Honest (this audit) |
|-------:|-----------------:|--------------------:|
| MAE    | \$866            | **\$38,248**        |
| MedAE  | —                | \$23,431            |
| RMSE   | —                | \$68,982            |
| MAPE   | 0.21%            | **7.57%**           |
| R²     | 0.99997          | **0.909**           |

That is a **~44× gap** between advertised and actual MAE. A plain,
leakage-free **Ridge regression beats the production model** on the same
hold-out (MAE \$32,726). See [Benchmark results](#benchmark-results).

---

## 1. Current architecture

| Aspect | Finding |
|---|---|
| Model type | `LGBMRegressor` (LightGBM 4.5.0), 3000 trees, `learning_rate=0.015`, `max_depth=10`, early stopping. Wrapped in a `sklearn` `Pipeline`/`ColumnTransformer` with `OneHotEncoder` for categoricals. |
| Training script | `ml/train_model.py` |
| Artifact | `trained_price_model.pkl` (plus ~70 timestamped `trained_price_model_*.pkl` snapshots checked into the repo root). |
| Loaded by | `app/streamlit_app.py:99` — `joblib.load("trained_price_model.pkl")`. |
| Data source | PostgreSQL `housing_data` table, via `utils/db_config.py` (`get_db_config`, env-var driven). |
| Target | `sold_price`. |
| Split | `train_test_split(test_size=0.2, stratify=<price quintiles>, random_state=42)` — **random, not chronological; no property/address grouping.** (`ml/train_model.py:151`) |
| Metrics reported | MAE, MAPE, R² on the random test split (`ml/train_model.py:194`). |
| Explainability | `model_explanation.json` written at train time and shown in the UI "Model Information" panel. |

### Data reality vs. the artifact
- The live DB `housing_data` table currently holds **4,364 rows**, all with
  `listing_date` in a single narrow window (**2026-02-28 → 2026-05-26**), and the
  values look synthetically generated (`data/generate_real_estate_data.py`).
- `model_explanation.json` reports `trained_on_rows = 20548`. The artifact was
  therefore trained on a **different (larger) dataset than what the DB now
  serves** — a train/serve data mismatch that by itself makes the reported
  metrics non-reproducible.

---

## 2. Target leakage (the core problem)

`ml/train_model.py` engineers three features from the target and then trains on
them:

```python
# ml/train_model.py:59-61
df['price_diff']      = df['sold_price'] - df['list_price']         # ← contains target
df['over_asking_pct'] = df['price_diff'] / df['list_price']         # ← contains target
df['price_per_sqft']  = df['sold_price'] / df['sqft']               # ← contains target
```

All three are in the training `features` list (`ml/train_model.py:132-140`), and
several downstream features are **derived from `price_per_sqft`**, so the leakage
spreads:

- `neighborhood_hotness` — from `groupby(neighborhood)[price_per_sqft].mean()` (`:98`)
- `realtor_logic` — weighted sum including `neighborhood_hotness` (`:102`)
- `neighborhood_ppsf_premium` — `neigh_median_ppsf / city_median_ppsf` (`:123`)
- `is_underpriced_for_bidding` — compares `list_price` to `neigh_median_ppsf` (`:87`)

`price_per_sqft = sold_price / sqft` is the most damaging: the model can recover
`sold_price ≈ price_per_sqft × sqft` almost exactly, which is exactly why R²
approaches 1.0 and MAE collapses to a few hundred dollars.

### Additional leakage / methodology issues
1. **Post-sale feature used as input:** `dom_days` (days on market) is only known
   *after* the listing sells. It drives `is_multi_offer_by_dom`,
   `is_not_multi_offer_by_dom`, `likely_multi_offer` (`:83-95`) — none are
   available at prediction time.
2. **Statistics computed on the full dataset before the split:** neighborhood
   medians/means, `city_median_ppsf`, `comp_count_in_neighborhood`, and the 99th
   percentile clips (`:75-126`) are all fit on **all** rows, then used for both
   train and test → validation-set information bleeds into training.
3. **Random split, not chronological:** future listings can train a model that is
   then "tested" on past ones. For a pricing model that will be used going
   forward, evaluation must be chronological.
4. **No duplicate-property separation:** the same address can appear in both train
   and test.

---

## 3. Train ⇄ inference feature mismatch

The UI cannot compute the leaky features (it does not know `sold_price`), so at
prediction time it feeds **placeholders** (`app/streamlit_app.py`, prediction
handler):

| Feature | Training value | Inference value (UI) |
|---|---|---|
| `price_diff` | `sold_price - list_price` | `0` |
| `over_asking_pct` | `(sold_price-list_price)/list_price` | `0` |
| `price_per_sqft` | `sold_price / sqft` | `list_price / sqft` |
| `neighborhood_hotness` | data-derived | `0.5` / `0.6` constant |
| `realtor_logic` | data-derived | `0.5` / `0.8` constant |

So the single most important learned signal (`price_per_sqft` built from
`sold_price`) is replaced at inference by a **different quantity** built from
`list_price`. The model effectively becomes "predict something close to
`list_price`," which is why the honest evaluation still gets a moderate R²
(list price is genuinely predictive) but a large RMSE and tail error.

**Consequence:** the model's real behaviour bears little relation to the
metrics on the "Model Information" card, and the card overstates accuracy by a
factor of ~44 on MAE.

---

## 4. Current metrics — honest re-evaluation

Method (`ml/benchmark_models.py`):
- Chronological split by `listing_date` (70% train / 15% val / 15% test),
  addresses held to a single split (4 duplicate-property rows removed).
- Rows with non-positive `sold_price`/`list_price`/`sqft` dropped → 4,249 usable
  rows (train 2,974 / val 637 / test 634).
- The production artifact scored with UI-style placeholder inputs.

| Model | MAE | MedAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| **production_model_honest** | **\$38,248** | \$23,431 | \$68,982 | 7.57% | 0.909 |
| ridge (leakage-free) | \$32,726 | \$24,586 | \$45,448 | 7.35% | 0.961 |
| random_forest (leakage-free) | \$32,729 | \$25,795 | \$45,561 | 7.07% | 0.960 |
| baseline: comparable \$/sqft | \$115,310 | \$77,722 | \$179,550 | 26.7% | 0.385 |
| baseline: global median | \$153,474 | \$110,000 | \$238,053 | 33.9% | −0.08 |

Where the current model hurts most (honest, by segment — see
`reports/model_segment_errors_*.csv`): high-value homes (**\$900k+ band MAE
≈ \$73k**), large homes (**2500+ sqft MAE ≈ \$74k**), and hot/expensive
neighborhoods (Tuxedo, Elmhurst, Crescentwood, Linden Woods).

---

## 5. Recommendations (summary — see `docs/model_v2_design.md`)

1. **Remove all target-derived and post-sale features.** Train only on
   information available before a sale (see v2 feature schema).
2. **Evaluate chronologically** with duplicate-property separation; treat
   `reports/model_benchmark_*.json` as the source of truth, not the training-set
   score.
3. **Fit neighborhood/market statistics inside the pipeline on train folds only.**
4. **Replace the single "winning offer" number** with three separate outputs
   (market value, expected sale-price range, offer strategy) plus a calibrated
   uncertainty interval.
5. **Stop committing ~70 model snapshots to the repo**; keep artifacts out of
   git and version them by metadata.
6. Do **not** trust or ship the current `model_explanation.json` metrics.

---

## Reproduce this audit

```bash
# Reads the live DB read-only through utils.db_config; writes only to reports/.
python -m ml.benchmark_models
```

(In this environment the deps live in the `samvisionai-samvision-app` image and
the DB is on the `samvision-net` docker network:)

```bash
docker run --rm --network samvision-net --env-file .env \
  -v "$PWD":/work -w /work samvisionai-samvision-app \
  python -m ml.benchmark_models
```
