import pandas as pd
import numpy as np
import psycopg2
import joblib
import ast
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load DB config
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

query = """
SELECT neighborhood, house_type, bedrooms, bathrooms, sqft,
       built_year, garage_type, season, sold_price, list_price, dom, listing_date
FROM housing_data;
"""

conn = psycopg2.connect(**db_config)
df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("[WARN] No data found.")
    joblib.dump(None, 'trained_price_model.pkl')
    exit()

df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
df['age'] = datetime.now().year - df['built_year']
df = df[(df['bathrooms'] > 0) & (df['bedrooms'] > 0) & (df['sqft'] > 300)]
df = df.dropna(subset=['sold_price', 'list_price', 'listing_date'])

# Recent listings boost
recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)

df['price_diff'] = df['sold_price'] - df['list_price']
df['over_asking_pct'] = df['price_diff'] / df['list_price']
df['price_per_sqft'] = df['sold_price'] / df['sqft']
df['dom_bucket'] = pd.cut(df['dom'], bins=[-1, 7, 14, 30, 90, 180], labels=['0-7', '8-14', '15-30', '31-90', '90+'])
df['multi_offer_flag'] = (df['over_asking_pct'] > 0.05).astype(int)

# Neighborhood hotness (faster DOM)
df['avg_dom'] = df.groupby("neighborhood")['dom'].transform("mean")
df['neighborhood_hotness'] = 1 - (df['avg_dom'] / df['avg_dom'].max())

# Realtor logic score
df['realtor_logic'] = (
    0.3 * (1 - df['age'] / 100) +
    0.4 * df['neighborhood_hotness'] +
    0.3 * df['multi_offer_flag']
)

# --- Select features
target = "sold_price"
features = [
    'bedrooms', 'bathrooms', 'sqft', 'age', 'dom', 'list_price',
    'price_diff', 'over_asking_pct', 'price_per_sqft', 'neighborhood_hotness',
    'realtor_logic', 'recency_weight',
    'house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood'
]

df = df[features + [target]].dropna()

X = df.drop(target, axis=1)
y = df[target]

categorical = ['house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood']
numerical = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"[OK] Trained | MAE=${mae:,.2f} | R²={r2:.3f}")
joblib.dump(pipeline, 'trained_price_model.pkl')
print(" Model saved to trained_price_model.pkl")
