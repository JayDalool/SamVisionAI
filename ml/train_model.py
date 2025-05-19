# file: ml/train_model.py

import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import ast
from datetime import datetime

# Load DB config
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

# Connect to DB and query updated fields
conn = psycopg2.connect(**db_config)
query = """
SELECT neighborhood, region, house_type, bedrooms, bathrooms, sqft, lot_size,
       built_year, garage_type, season, sold_price
FROM housing_data;
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Calculate age from built_year
df['age'] = datetime.now().year - df['built_year']

# One-hot encode categorical variables
categorical_cols = ['neighborhood', 'region', 'house_type', 'season', 'garage_type']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and target
X = df.drop(['sold_price', 'built_year'], axis=1)
y = df['sold_price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"✅ Model trained: MAE=${mae:.2f}, R^2={r2:.2f}")

# Save model
joblib.dump(model, 'trained_price_model.pkl')
print("✅ Model saved to 'trained_price_model.pkl'")

import matplotlib.pyplot as plt

# Feature importance
importances = model.feature_importances_
features = X.columns

# Sort for better visual
sorted_idx = importances.argsort()[::-1]
sorted_features = features[sorted_idx]
sorted_importances = importances[sorted_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:20], sorted_importances[:20])
plt.xlabel("Feature Importance")
plt.title("Top 20 Most Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 Feature importance chart saved as 'feature_importance.png'")
