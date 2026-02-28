# file: ml/train_model.py

import pandas as pd
import numpy as np
import joblib
import ast
import os
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.db_config import get_db_config
from sqlalchemy import create_engine
print("🚀 Starting SamVision AI model training pipeline with Realtor Multi-Offer Logic...")

# ========================
# DB Config & Load
# ========================
cfg = get_db_config()
engine = create_engine(
    f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
)

query = """
SELECT neighborhood, house_type, style, bedrooms, bathrooms, sqft,
       built_year, garage_type, season, sold_price, list_price, listing_date, dom_days
FROM housing_data;
"""

df = pd.read_sql_query(query, engine)

if df.empty:
    print("[⚠️] No data found in DB. Exiting.")
    joblib.dump(None, 'trained_price_model.pkl')
    exit()

print(f"[ℹ️] Loaded {len(df)} rows from DB")

# ========================
# Preprocessing & Feature Engineering
# ========================
df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
df['age'] = datetime.now().year - df['built_year']
df['price_diff'] = df['sold_price'] - df['list_price']
df['over_asking_pct'] = df['price_diff'] / df['list_price']
df['price_per_sqft'] = df['sold_price'] / df['sqft']

print(f"[ℹ️] Rows before filters: {len(df)}")
df = df[
    (df['bathrooms'] > 0) &
    (df['bedrooms'] > 0) &
    (df['sqft'] >= 500) &
    (df['sold_price'] > 0) &
    (df['list_price'] > 0) &
    (df['dom_days'] >= 0)
].dropna(subset=['sold_price', 'list_price', 'listing_date'])
print(f"[ℹ️] Rows after filtering: {len(df)}")

# Outlier capping
df['sold_price'] = df['sold_price'].clip(upper=df['sold_price'].quantile(0.99))
df['list_price'] = df['list_price'].clip(upper=df['list_price'].quantile(0.99))

# Recency weighting
recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)

# Multi-offer Realtor logic
df['is_multi_offer_by_dom'] = (df['dom_days'] <= 3).astype(int)
df['is_not_multi_offer_by_dom'] = (df['dom_days'] > 11).astype(int)

neigh_median_ppsf = df.groupby('neighborhood')['price_per_sqft'].transform('median')
df['is_underpriced_for_bidding'] = (
    (df['list_price'] < neigh_median_ppsf * df['sqft'] * 0.95) &
    (df['is_multi_offer_by_dom'] == 1)
).astype(int)

df['likely_multi_offer'] = (
    df['is_multi_offer_by_dom'] |
    df['is_underpriced_for_bidding']
).astype(int)

# Neighborhood hotness
df['neighborhood_hotness'] = 1 - (
    df.groupby('neighborhood')['price_per_sqft'].transform('mean') / df['price_per_sqft'].max()
)

df['realtor_logic'] = (
    0.4 * df['neighborhood_hotness'] +
    0.3 * df['likely_multi_offer'] +
    0.3 * (1 - df['age'] / 100)
)

# Hot areas
hot_areas = [
    "Bridgwater", "Prairie Pointe", "Whyte Ridge", "East Transcona", "River Heights",
    "St. Vital", "Tuxedo", "North Kildonan", "Osborne Village", "Fort Richmond",
    "Bridgwater Trails", "Bridgwater Lakes", "Bridgwater Forest", "Bridgwater Centre"
]
df['is_hot_area'] = df['neighborhood'].isin(hot_areas).astype(int)

# Season boost
df['season_boost'] = df['season'].map({
    'Spring': 1.1, 'Summer': 1.05, 'Fall': 1.0, 'Winter': 0.95
}).fillna(1.0)

# Neighborhood premium
city_median_ppsf = df['price_per_sqft'].median()
df['neighborhood_ppsf_premium'] = neigh_median_ppsf / city_median_ppsf

# Comp count
df['comp_count_in_neighborhood'] = df['neighborhood'].map(df['neighborhood'].value_counts())

# ========================
# Features & Target
# ========================
target = "sold_price"
features = [
    'bedrooms', 'bathrooms', 'sqft', 'age', 'list_price',
    'price_diff', 'over_asking_pct', 'price_per_sqft',
    'neighborhood_hotness', 'realtor_logic', 'recency_weight',
    'likely_multi_offer', 'is_multi_offer_by_dom', 'is_not_multi_offer_by_dom', 'is_underpriced_for_bidding',
    'is_hot_area', 'neighborhood_ppsf_premium',
    'season_boost', 'comp_count_in_neighborhood',
    'house_type', 'garage_type', 'season', 'neighborhood', 'style'
]

df = df[features + [target]].dropna()
print(f"[ℹ️] Final training set size: {len(df)} rows")

X = df.drop(columns=[target])
y = df[target]

# ========================
# Train/Test Split
# ========================
bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=bins, random_state=42
)

categorical = ['house_type', 'garage_type', 'season', 'neighborhood', 'style']
numerical = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
], remainder='passthrough')

# ========================
# Model
# ========================
model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.015,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# ========================
# Training
# ========================
pipeline.named_steps['model'].fit(
    preprocessor.fit_transform(X_train),
    y_train,
    eval_set=[(preprocessor.transform(X_test), y_test)],
    eval_metric='mae',
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=50)]
)

y_pred = pipeline.named_steps['model'].predict(preprocessor.transform(X_test))
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"[✅] Model trained | MAE=${mae:,.0f} | MAPE={mape:.2%} | R²={r2:.4f}")

# ========================
# Save Model & Explanation
# ========================
import json

explanation_payload = {
    "model_used": "LightGBMRegressor with Realtor Multi-Offer Intelligence",
    "trained_on_rows": len(df),
    "mae": float(mae),
    "mape": float(mape),
    "r2": float(r2),
    "features_used": features,
    "note": (
        "This model incorporates Realtor knowledge of multi-offer vs. non-multi-offer using DOM, "
        "underpricing strategy detection, neighborhood premium analysis, and seasonality adjustments for price prediction."
    )
}

with open("model_explanation.json", "w") as f:
    json.dump(explanation_payload, f, indent=4)

print("[💡] model_explanation.json saved for Streamlit UI explainability.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"trained_price_model_{timestamp}.pkl"
joblib.dump(pipeline, model_filename)
joblib.dump(pipeline, "trained_price_model.pkl")
print(f"[💾] Model saved as {model_filename} and updated trained_price_model.pkl")

# ========================
# Log Training
# ========================
log_entry = {
    "timestamp": timestamp,
    "model_path": model_filename,
    "mae": mae,
    "mape": mape,
    "r2": r2,
    "rows_trained_on": len(df),
    "features": features
}

pd.DataFrame([log_entry]).to_csv("model_training_logs.csv", mode='a', header=not os.path.exists("model_training_logs.csv"), index=False)
print("[🗂️] Training log updated in model_training_logs.csv")

