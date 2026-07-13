# file: ml/benchmark_models.py
"""Reproducible, leakage-free benchmark of candidate valuation models.

Run from the repo root:

    python -m ml.benchmark_models

What it does
------------
* Loads housing data through the existing central DB config (utils.db_config).
* Validates required columns and cleans obviously-invalid rows.
* Uses ONLY information available *before* a sale as features (no sold_price,
  no price_per_sqft, no over-asking, no sell/list ratio, no days-on-market).
* Splits chronologically by the real ``sold_date`` (from the WRREB reports;
  train -> validation -> test) and removes any address that would otherwise
  appear in more than one split, so a single property can't leak across the
  split boundary.
* Benchmarks several candidate models plus simple baselines, all wrapped in a
  single sklearn Pipeline so preprocessing is identical for train and inference.
* Also evaluates the *current production model* honestly, feeding it the same
  inputs the live UI feeds, to quantify the leakage gap.
* Writes timestamped reports to ``reports/`` and NEVER touches the production
  model artifact.

It is intentionally read-only with respect to the database and the production
model; it only writes new files under ``reports/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.db_config import get_db_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
PRODUCTION_MODEL_PATH = "trained_price_model.pkl"
REPORTS_DIR = "reports"

TARGET = "sold_price"

# Pre-sale feature schema. Everything here is knowable *before* the property
# sells. Engineered columns (age, sold_month) are derived below.
NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft",
    "lot_size",
    "built_year",
    "age",
    "list_price",
    "sold_month",
]
CATEGORICAL_FEATURES = [
    "neighborhood",
    "house_type",
    "style",
    "garage_type",
    "basement_type",
    "season",
]

# Columns that must exist in the source table for the benchmark to be valid.
REQUIRED_COLUMNS = {
    "neighborhood", "house_type", "style", "garage_type", "basement_type",
    "bedrooms", "bathrooms", "sqft", "built_year", "list_price", "sold_price",
    "sold_date", "season", "address",
}

# Columns that must NOT be used as features because they leak the target or are
# only known after the sale. Documented here so the exclusion is explicit.
LEAKING_COLUMNS = {
    "sold_price", "sell_list_ratio", "dom_days",
    # derived-at-inference leaks used by the legacy model:
    "price_diff", "over_asking_pct", "price_per_sqft",
}


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
def get_engine():
    """Create a SQLAlchemy engine from the central DB config (no secrets logged)."""
    cfg = get_db_config()
    url = URL.create(
        "postgresql+psycopg2",
        username=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["dbname"],
    )
    return create_engine(url)


def load_housing_data(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    df = pd.read_sql_query("SELECT * FROM housing_data", engine)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "housing_data is missing required columns: " + ", ".join(sorted(missing))
        )


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows and derive leakage-free engineered features.

    Time features and the chronological split come from the real ``sold_date``
    (WRREB canonical field). ``sale_year``/``sale_month`` are derived from it;
    the MLS-embedded year is never used as a temporal truth. There is no
    fabricated ``listing_month``.
    """
    df = df.copy()
    df["sold_date"] = pd.to_datetime(df["sold_date"], errors="coerce")

    df = df[
        (df["sold_price"] > 0)
        & (df["list_price"] > 0)
        & (df["sqft"] > 0)
    ].dropna(subset=["sold_date", "sold_price", "list_price", "sqft"])

    df["age"] = (df["sold_date"].dt.year - df["built_year"]).clip(lower=0)
    df["sold_month"] = df["sold_date"].dt.month

    if "lot_size" not in df.columns:
        df["lot_size"] = np.nan

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("object").where(df[col].notna(), "Unknown")

    df = df.sort_values("sold_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Chronological, duplicate-property-safe split
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    dropped_for_dup_property: int


def chronological_group_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> SplitResult:
    """Split by time, then guarantee no address spans two splits.

    Rows are ordered by ``sold_date`` (already sorted upstream). The earliest
    ``train_frac`` become training, the next ``val_frac`` validation, and the
    remainder test. Any validation/test row whose address already appears in an
    earlier split is dropped so a single property never leaks across the
    boundary.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    train_addr = set(train["address"])
    before = len(val) + len(test)
    val = val[~val["address"].isin(train_addr)].copy()
    seen = train_addr | set(val["address"])
    test = test[~test["address"].isin(seen)].copy()
    dropped = before - (len(val) + len(test))

    return SplitResult(train=train, val=val, test=test, dropped_for_dup_property=dropped)


# ---------------------------------------------------------------------------
# Preprocessing / model pipelines
# ---------------------------------------------------------------------------
def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def candidate_pipelines() -> dict[str, Pipeline]:
    """sklearn Pipelines (identical preprocessing for train + inference)."""
    return {
        "ridge": Pipeline(
            [("prep", _preprocessor(scale_numeric=True)), ("model", Ridge(alpha=10.0, random_state=RANDOM_SEED))]
        ),
        "random_forest": Pipeline(
            [
                ("prep", _preprocessor(scale_numeric=False)),
                ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_SEED)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("prep", _preprocessor(scale_numeric=False)),
                ("model", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_SEED)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(scale_numeric=False)),
                ("model", HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600, l2_regularization=1.0, random_state=RANDOM_SEED)),
            ]
        ),
    }


def _maybe_lightgbm() -> dict[str, Pipeline]:
    """LightGBM only if it is already installed (it is, per requirements.txt)."""
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return {}
    return {
        "lightgbm": Pipeline(
            [
                ("prep", _preprocessor(scale_numeric=False)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=800,
                        learning_rate=0.03,
                        num_leaves=63,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    }


# ---------------------------------------------------------------------------
# Baselines (fit on train only)
# ---------------------------------------------------------------------------
@dataclass
class BaselinePredictor:
    name: str
    _predict: Callable[[pd.DataFrame], np.ndarray]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(X)


def build_baselines(train: pd.DataFrame) -> dict[str, BaselinePredictor]:
    global_median = float(train[TARGET].median())
    global_ppsf = float((train[TARGET] / train["sqft"]).median())
    hood_ppsf = (train[TARGET] / train["sqft"]).groupby(train["neighborhood"]).median()

    def global_median_pred(X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), global_median)

    def comparable_ppsf_pred(X: pd.DataFrame) -> np.ndarray:
        ppsf = X["neighborhood"].map(hood_ppsf).fillna(global_ppsf).to_numpy()
        return ppsf * X["sqft"].to_numpy()

    return {
        "baseline_global_median": BaselinePredictor("baseline_global_median", global_median_pred),
        "baseline_comparable_ppsf": BaselinePredictor("baseline_comparable_ppsf", comparable_ppsf_pred),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) if mask.any() else float("nan")
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    return {"n": int(len(y_true)), "mae": mae, "medae": medae, "rmse": rmse, "mape": mape, "r2": r2}


def _sqft_band(sqft: float) -> str:
    for hi, label in [(900, "<900"), (1300, "900-1299"), (1800, "1300-1799"), (2500, "1800-2499")]:
        if sqft < hi:
            return label
    return "2500+"


def _price_band(price: float) -> str:
    for hi, label in [(300_000, "<300k"), (450_000, "300-450k"), (600_000, "450-600k"), (900_000, "600-900k")]:
        if price < hi:
            return label
    return "900k+"


def segment_errors(model_name: str, test: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    df = test.copy()
    df["_pred"] = y_pred
    df["_abs_err"] = (df[TARGET] - df["_pred"]).abs()
    df["price_band"] = df[TARGET].map(_price_band)
    df["sqft_band"] = df["sqft"].map(_sqft_band)

    rows = []
    for seg_type, col in [
        ("neighborhood", "neighborhood"),
        ("house_type", "house_type"),
        ("price_band", "price_band"),
        ("sqft_band", "sqft_band"),
    ]:
        grouped = df.groupby(col)
        for value, g in grouped:
            m = compute_metrics(g[TARGET].to_numpy(), g["_pred"].to_numpy())
            rows.append(
                {
                    "model": model_name,
                    "segment_type": seg_type,
                    "segment_value": value,
                    "n": m["n"],
                    "mae": round(m["mae"], 2),
                    "medae": round(m["medae"], 2),
                    "mape": round(m["mape"], 4),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Honest evaluation of the current production model
# ---------------------------------------------------------------------------
def evaluate_production_model(test: pd.DataFrame, full: pd.DataFrame, model_path: str) -> dict | None:
    """Score the live production model the way the UI actually calls it.

    The production model was trained on leaky features (price_per_sqft from
    sold_price, price_diff, over_asking_pct). At inference the UI cannot supply
    those, so it passes neutral placeholders (0 / list_price-based). Feeding the
    same placeholders here reveals the model's *real* out-of-sample error.
    """
    import joblib

    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        if model is None:
            return None

        comp_count = full["neighborhood"].map(full["neighborhood"].value_counts())
        comp_lookup = dict(zip(full["neighborhood"], comp_count))

        rows = []
        for _, r in test.iterrows():
            age = int(r["age"])
            list_price = float(r["list_price"])
            sqft = float(r["sqft"])
            rows.append(
                {
                    "bedrooms": r["bedrooms"], "bathrooms": r["bathrooms"], "sqft": sqft,
                    "built_year": r["built_year"], "age": age, "list_price": list_price,
                    "price_diff": 0, "over_asking_pct": 0,
                    "price_per_sqft": list_price / max(sqft, 1),
                    "neighborhood_hotness": 0.5, "realtor_logic": 0.5, "recency_weight": 1,
                    "multi_offer_flag": 0, "likely_multi_offer": 0, "season_boost": 1.0,
                    "comp_count_in_neighborhood": comp_lookup.get(r["neighborhood"], 0),
                    "house_type": r["house_type"], "garage_type": r["garage_type"],
                    "season": r["season"], "neighborhood": r["neighborhood"], "style": r["style"],
                }
            )
        input_df = pd.DataFrame(rows)

        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[list(model.feature_names_in_)].fillna(0)

        y_pred = model.predict(input_df)
        metrics = compute_metrics(test[TARGET].to_numpy(), y_pred)
        metrics["_y_pred"] = y_pred  # internal, stripped before JSON dump
        return metrics
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_benchmark(output_dir: str = REPORTS_DIR) -> dict:
    np.random.seed(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw = load_housing_data()
    validate_columns(raw)
    df = clean_and_engineer(raw)
    split = chronological_group_split(df)

    train, val, test = split.train, split.val, split.test
    # Train on train+val (val used only to confirm the split; models here are not
    # early-stopped, so folding val into fit is fine and gives more signal).
    fit_df = pd.concat([train, val], ignore_index=True)
    X_fit, y_fit = fit_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], fit_df[TARGET]
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[TARGET].to_numpy()

    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    # Baselines
    for name, base in build_baselines(fit_df).items():
        pred = base.predict(test)
        predictions[name] = pred
        results[name] = compute_metrics(y_test, pred)

    # Candidate models
    models = {**candidate_pipelines(), **_maybe_lightgbm()}
    for name, pipe in models.items():
        pipe.fit(X_fit, y_fit)
        pred = pipe.predict(X_test)
        predictions[name] = pred
        results[name] = compute_metrics(y_test, pred)

    # Production model (honest)
    prod = evaluate_production_model(test, df, PRODUCTION_MODEL_PATH)
    if prod is not None and "error" not in prod:
        predictions["production_model_honest"] = prod.pop("_y_pred")
        results["production_model_honest"] = prod
    elif prod is not None:
        results["production_model_honest"] = prod

    # Rank by test MAE (lower is better) among scored models
    scored = {k: v for k, v in results.items() if "mae" in v}
    best_name = min(scored, key=lambda k: scored[k]["mae"])

    # Segment errors for the best candidate + production model (when available)
    seg_frames = [segment_errors(best_name, test, predictions[best_name])]
    if "production_model_honest" in predictions:
        seg_frames.append(segment_errors("production_model_honest", test, predictions["production_model_honest"]))
    segments = pd.concat(seg_frames, ignore_index=True)

    report = {
        "timestamp": timestamp,
        "random_seed": RANDOM_SEED,
        "rows_total": int(len(df)),
        "split": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
            "dropped_for_duplicate_property": int(split.dropped_for_dup_property),
            "strategy": "chronological by real sold_date, address held to a single split",
            "train_date_range": [str(train["sold_date"].min().date()), str(train["sold_date"].max().date())],
            "test_date_range": [str(test["sold_date"].min().date()), str(test["sold_date"].max().date())],
        },
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "excluded_leaking_columns": sorted(LEAKING_COLUMNS),
        "best_candidate": best_name,
        "metrics": results,
    }

    json_path = os.path.join(output_dir, f"model_benchmark_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"model_benchmark_{timestamp}.csv")
    seg_path = os.path.join(output_dir, f"model_segment_errors_{timestamp}.csv")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    metrics_rows = [{"model": k, **v} for k, v in results.items()]
    pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
    segments.to_csv(seg_path, index=False)

    print(f"[✅] Benchmark complete. Best candidate by test MAE: {best_name}")
    print(f"     rows={len(df)} train={len(train)} val={len(val)} test={len(test)} "
          f"(dropped {split.dropped_for_dup_property} dup-property rows)")
    for name in sorted(scored, key=lambda k: scored[k]["mae"]):
        m = scored[name]
        print(f"     {name:26s} MAE=${m['mae']:>10,.0f}  MedAE=${m['medae']:>10,.0f}  "
              f"RMSE=${m['rmse']:>10,.0f}  MAPE={m['mape']:.2%}  R2={m['r2']:.4f}")
    print(f"[💾] Wrote:\n     {json_path}\n     {csv_path}\n     {seg_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate valuation models (leakage-free).")
    parser.add_argument("--output-dir", default=REPORTS_DIR, help="Directory for report files.")
    args = parser.parse_args()
    run_benchmark(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
