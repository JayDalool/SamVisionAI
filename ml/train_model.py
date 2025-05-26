# file: ml/train_model.py

import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import ast
from datetime import datetime

# --- Load DB credentials
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

# --- Query Real Data from PostgreSQL
query = """
SELECT neighborhood, region, house_type, bedrooms, bathrooms, sqft, lot_size,
       built_year, garage_type, season, sold_price, listing_date,
       address, dom, latitude, longitude
FROM housing_data;
"""

conn = psycopg2.connect(**db_config)
df = pd.read_sql_query(query, conn)
conn.close()

# --- Feature Engineering
df['age'] = datetime.now().year - df['built_year']
df['dom'] = df['dom'].fillna((pd.Timestamp.now() - pd.to_datetime(df['listing_date'])).dt.days.clip(lower=0))

# --- One-Hot Encode Categorical
categorical_cols = ['neighborhood', 'region', 'house_type', 'season', 'garage_type']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- Train/Test Split
X = df.drop(['sold_price', 'built_year', 'listing_date', 'address'], axis=1)
y = df['sold_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Regressor
model = RandomForestRegressor(n_estimators=120, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f" Model trained | MAE=${mae:,.2f} | R²={r2:.2f}")

# --- Save model
joblib.dump(model, 'trained_price_model.pkl')
print(" Model saved to 'trained_price_model.pkl'")
