# # file: ml/train_model.py

# import pandas as pd
# import numpy as np
# import psycopg2
# import joblib
# import ast
# from datetime import datetime
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# # Load DB config
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# query = """
# SELECT neighborhood, house_type, bedrooms, bathrooms, sqft,
#        built_year, garage_type, season, sold_price, list_price, dom
# FROM housing_data;
# """

# conn = psycopg2.connect(**db_config)
# df = pd.read_sql_query(query, conn)
# conn.close()

# if df.empty:
#     print("[WARN] No data found.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# # Clean & derive age
# current_year = datetime.now().year
# df['age'] = current_year - df['built_year']
# df = df[(df['bathrooms'] > 0) & (df['bedrooms'] > 0) & (df['sqft'] > 300)]
# df = df.dropna(subset=['sold_price', 'list_price'])

# # Final training fields
# features = [
#     'bedrooms', 'bathrooms', 'sqft', 'age', 'dom', 'list_price',
#     'house_type', 'garage_type', 'season', 'neighborhood'
# ]
# target = 'sold_price'

# # Drop any remaining nulls
# df = df[features + [target]].dropna()
# X = df[features]
# y = df[target]

# # Preprocess
# categorical = ['house_type', 'garage_type', 'season', 'neighborhood']
# numerical = [col for col in X.columns if col not in categorical]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
# ], remainder='passthrough')

# pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
# ])

# # Train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# pipeline.fit(X_train, y_train)

# # Evaluate
# y_pred = pipeline.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"[OK] Trained | MAE=${mae:,.2f} | R²={r2:.3f}")
# joblib.dump(pipeline, 'trained_price_model.pkl')
# print(" Model saved to trained_price_model.pkl")
# # file: ml/train_model.py

# import pandas as pd
# import numpy as np
# import psycopg2
# import joblib
# import ast
# from datetime import datetime
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# # Load DB config
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# query = """
# SELECT neighborhood, house_type, bedrooms, bathrooms, sqft,
#        built_year, garage_type, season, sold_price, list_price, dom, listing_date
# FROM housing_data;
# """

# conn = psycopg2.connect(**db_config)
# df = pd.read_sql_query(query, conn)
# conn.close()

# if df.empty:
#     print("[WARN] No data found in DB.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# # Preprocess & filters
# df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
# df['age'] = datetime.now().year - df['built_year']

# # Drop missing or clearly invalid values
# df = df[(df['bathrooms'] > 0) & (df['bedrooms'] > 0) & (df['sqft'] > 300)]
# df = df.dropna(subset=['sold_price', 'list_price', 'listing_date'])
# df = df[(df['sold_price'] > 0) & (df['list_price'] > 0)]

# if df.empty:
#     print("[FAIL] No valid rows with non-zero sold_price/list_price.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# # Feature engineering
# recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
# df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)

# df['price_diff'] = df['sold_price'] - df['list_price']
# df['over_asking_pct'] = df['price_diff'] / df['list_price']
# df['price_per_sqft'] = df['sold_price'] / df['sqft']
# df['dom_bucket'] = pd.cut(df['dom'], bins=[-1, 7, 14, 30, 90, 180], labels=['0-7', '8-14', '15-30', '31-90', '90+'])
# df['multi_offer_flag'] = (df['over_asking_pct'] > 0.05).astype(int)

# df['avg_dom'] = df.groupby("neighborhood")['dom'].transform("mean")
# df['neighborhood_hotness'] = 1 - (df['avg_dom'] / df['avg_dom'].max())

# df['realtor_logic'] = (
#     0.3 * (1 - df['age'] / 100) +
#     0.4 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag']
# )

# # Model features
# target = "sold_price"
# features = [
#     'bedrooms', 'bathrooms', 'sqft', 'age', 'dom', 'list_price',
#     'price_diff', 'over_asking_pct', 'price_per_sqft', 'neighborhood_hotness',
#     'realtor_logic', 'recency_weight',
#     'house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood'
# ]

# df = df[features + [target]].dropna()

# if df.empty:
#     print("[FAIL] Data cleaned to empty — check for broken fields.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# X = df.drop(target, axis=1)
# y = df[target]

# categorical = ['house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood']
# numerical = [col for col in X.columns if col not in categorical]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
# ], remainder='passthrough')

# pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
# ])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"[OK] Trained | MAE=${mae:,.2f} | R²={r2:.3f}")
# joblib.dump(pipeline, 'trained_price_model.pkl')
# print("✅ Model saved to trained_price_model.pkl")

# file: ml/train_model.py

# import pandas as pd
# import numpy as np
# import psycopg2
# import joblib
# import ast
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.metrics import mean_absolute_error, r2_score

# # Load DB config
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# query = """
# SELECT neighborhood, house_type, bedrooms, bathrooms, sqft,
#        built_year, garage_type, season, sold_price, list_price, dom, listing_date
# FROM housing_data;
# """

# conn = psycopg2.connect(**db_config)
# df = pd.read_sql_query(query, conn)
# conn.close()

# if df.empty:
#     print("[WARN] No data found in DB.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# # Preprocess
# df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
# df['age'] = datetime.now().year - df['built_year']

# df = df[(df['bathrooms'] > 0) & (df['bedrooms'] > 0) & (df['sqft'] > 300)]
# df = df.dropna(subset=['sold_price', 'list_price', 'listing_date'])
# df = df[(df['sold_price'] > 0) & (df['list_price'] > 0)]

# if df.empty:
#     print("[FAIL] No valid rows with non-zero sold_price/list_price.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
# df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)
# df['price_diff'] = df['sold_price'] - df['list_price']
# df['over_asking_pct'] = df['price_diff'] / df['list_price']
# df['price_per_sqft'] = df['sold_price'] / df['sqft']
# df['dom_bucket'] = pd.cut(df['dom'], bins=[-1, 7, 14, 30, 90, 180],
#                           labels=['0-7', '8-14', '15-30', '31-90', '90+'])
# df['multi_offer_flag'] = (df['over_asking_pct'] > 0.05).astype(int)

# df['avg_dom'] = df.groupby("neighborhood")['dom'].transform("mean")
# df['neighborhood_hotness'] = 1 - (df['avg_dom'] / df['avg_dom'].max())

# df['realtor_logic'] = (
#     0.3 * (1 - df['age'] / 100) +
#     0.4 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag']
# )

# target = "sold_price"
# features = [
#     'bedrooms', 'bathrooms', 'sqft', 'age', 'dom', 'list_price',
#     'price_diff', 'over_asking_pct', 'price_per_sqft', 'neighborhood_hotness',
#     'realtor_logic', 'recency_weight',
#     'house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood'
# ]

# df = df[features + [target]].dropna()
# if df.empty:
#     print("[FAIL] Data cleaned to empty — check for broken fields.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# X = df.drop(target, axis=1)
# y = df[target]

# categorical = ['house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood']
# numerical = [col for col in X.columns if col not in categorical]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
# ], remainder='passthrough')

# pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", LGBMRegressor(
#         n_estimators=2000,
#         learning_rate=0.05,
#         max_depth=8,
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Fit with callbacks for early stopping + logging
# pipeline.named_steps['model'].fit(
#     preprocessor.fit_transform(X_train), y_train,
#     eval_set=[(preprocessor.transform(X_test), y_test)],
#     eval_metric='mae',
#     callbacks=[
#         early_stopping(stopping_rounds=50),
#         log_evaluation(period=50)
#     ]
# )

# y_pred = pipeline.named_steps['model'].predict(preprocessor.transform(X_test))

# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"[OK] Trained LGBM | MAE=${mae:,.2f} | R²={r2:.3f}")
# joblib.dump(pipeline, 'trained_price_model.pkl')
# print("✅ Model saved to trained_price_model.pkl")

# # file: ml/train_model.py

# import pandas as pd
# import numpy as np
# import joblib
# import ast
# from datetime import datetime
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.metrics import mean_absolute_error, r2_score

# # Load DB config using SQLAlchemy
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )

# query = """
# SELECT neighborhood, house_type, bedrooms, bathrooms, sqft,
#        built_year, garage_type, season, sold_price, list_price, dom, listing_date
# FROM housing_data;
# """

# df = pd.read_sql_query(query, engine)

# if df.empty:
#     print("[WARN] No data found in DB.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# # =========================
# # Data Preprocessing
# # =========================

# df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
# df['age'] = datetime.now().year - df['built_year']

# df = df[(df['bathrooms'] > 0) & (df['bedrooms'] > 0) & (df['sqft'] > 300)]
# df = df.dropna(subset=['sold_price', 'list_price', 'listing_date'])
# df = df[(df['sold_price'] > 0) & (df['list_price'] > 0)]

# recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
# df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)
# df['price_diff'] = df['sold_price'] - df['list_price']
# df['over_asking_pct'] = df['price_diff'] / df['list_price']
# df['price_per_sqft'] = df['sold_price'] / df['sqft']
# df['dom_bucket'] = pd.cut(df['dom'], bins=[-1, 7, 14, 30, 90, 180],
#                           labels=['0-7', '8-14', '15-30', '31-90', '90+'])
# df['multi_offer_flag'] = (df['over_asking_pct'] > 0.05).astype(int)
# df['avg_dom'] = df.groupby("neighborhood")['dom'].transform("mean")
# df['neighborhood_hotness'] = 1 - (df['avg_dom'] / df['avg_dom'].max())
# df['realtor_logic'] = (
#     0.3 * (1 - df['age'] / 100) +
#     0.4 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag']
# )

# # =========================
# # Features & Target
# # =========================

# target = "sold_price"
# features = [
#     'bedrooms', 'bathrooms', 'sqft', 'age', 'dom', 'list_price',
#     'price_diff', 'over_asking_pct', 'price_per_sqft', 'neighborhood_hotness',
#     'realtor_logic', 'recency_weight',
#     'house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood'
# ]

# df = df[features + [target]].dropna()
# if df.empty:
#     print("[FAIL] Cleaned data is empty. Check input data.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# X = df.drop(target, axis=1)
# y = df[target]

# # Stratified splitting using price bins to improve train-test distribution across price ranges
# bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=bins, random_state=42
# )

# categorical = ['house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood']
# numerical = [col for col in X.columns if col not in categorical]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
# ], remainder='passthrough')

# model = LGBMRegressor(
#     n_estimators=2000,
#     learning_rate=0.03,
#     max_depth=10,
#     subsample=0.9,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )

# pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", model)
# ])

# # Fit model with callbacks
# pipeline.named_steps['model'].fit(
#     preprocessor.fit_transform(X_train),
#     y_train,
#     eval_set=[(preprocessor.transform(X_test), y_test)],
#     eval_metric='mae',
#     callbacks=[
#         early_stopping(stopping_rounds=50),
#         log_evaluation(period=50)
#     ]
# )

# y_pred = pipeline.named_steps['model'].predict(preprocessor.transform(X_test))

# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"[✅] LGBM Model Trained | MAE=${mae:,.2f} | R²={r2:.4f}")

# # Save with timestamped version for tracking experiments
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_path = f"trained_price_model_{timestamp}.pkl"
# joblib.dump(pipeline, model_path)
# joblib.dump(pipeline, "trained_price_model.pkl")

# print(f"✅ Model saved to {model_path} and updated trained_price_model.pkl")

# # file: ml/train_model.py
# import pandas as pd
# import numpy as np
# import joblib
# import ast
# from datetime import datetime
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# print("🚀 Starting model training pipeline with HOT AREAS enhancements...")

# # ========================
# # DB Config and Load
# # ========================
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )

# query = """
# SELECT neighborhood, house_type, bedrooms, bathrooms, sqft,
#        built_year, garage_type, season, sold_price, list_price, dom, listing_date
# FROM housing_data;
# """

# df = pd.read_sql_query(query, engine)

# if df.empty:
#     print("[⚠️] No data found in DB.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# # ========================
# # Preprocessing
# # ========================
# df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
# df['age'] = datetime.now().year - df['built_year']
# df = df[(df['bathrooms'] > 0) & (df['bedrooms'] > 0) & (df['sqft'] > 300)]
# df = df.dropna(subset=['sold_price', 'list_price', 'listing_date'])
# df = df[(df['sold_price'] > 0) & (df['list_price'] > 0)]

# # Outlier clipping
# df['sold_price'] = df['sold_price'].clip(upper=df['sold_price'].quantile(0.99))
# df['list_price'] = df['list_price'].clip(upper=df['list_price'].quantile(0.99))

# # Market heat features
# recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
# df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)
# df['price_diff'] = df['sold_price'] - df['list_price']
# df['over_asking_pct'] = df['price_diff'] / df['list_price']
# df['price_per_sqft'] = df['sold_price'] / df['sqft']
# df['dom_bucket'] = pd.cut(df['dom'], bins=[-1, 7, 14, 30, 90, 180],
#                           labels=['0-7', '8-14', '15-30', '31-90', '90+'])
# df['multi_offer_flag'] = (df['over_asking_pct'] > 0.05).astype(int)
# df['avg_dom_neigh'] = df.groupby("neighborhood")['dom'].transform("mean")
# df['neighborhood_hotness'] = 1 - (df['avg_dom_neigh'] / df['avg_dom_neigh'].max())

# # Market heat composite
# df['market_heat_score'] = (
#     0.4 * df['recency_weight'] +
#     0.3 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag']
# )

# # Realtor logic feature
# df['realtor_logic'] = (
#     0.3 * (1 - df['age'] / 100) +
#     0.4 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag']
# )

# # ========================
# # Add HOT AREAS Flags
# # ========================
# hot_areas = [
#     "Bridgwater", "Prairie Pointe", "Whyte Ridge", "East Transcona",
#     "River Heights", "St. Vital", "Tuxedo", "North Kildonan",
#     "Osborne Village", "Fort Richmond", "Bridgwater Trails",
#     "Bridgwater Lakes", "Bridgwater Forest", "Bridgwater Centre"
# ]
# df['is_hot_area'] = df['neighborhood'].isin(hot_areas).astype(int)

# # Neighborhood PPSF Premium
# neigh_median_ppsf = df.groupby('neighborhood')['price_per_sqft'].transform('median')
# city_median_ppsf = df['price_per_sqft'].median()
# df['neighborhood_ppsf_premium'] = neigh_median_ppsf / city_median_ppsf

# # Likely multi-offer engineered feature
# df['likely_multi_offer'] = (
#     (df['list_price'] < neigh_median_ppsf * df['sqft'] * 0.98) &
#     (df['market_heat_score'] > 0.6)
# ).astype(int)

# # Season weighted boost
# df['season_boost'] = df['season'].map({
#     'Spring': 1.1,
#     'Summer': 1.05,
#     'Fall': 1.0,
#     'Winter': 0.95
# }).fillna(1.0)

# # ========================
# # Features and Target
# # ========================
# target = "sold_price"
# features = [
#     'bedrooms', 'bathrooms', 'sqft', 'age', 'dom', 'list_price',
#     'price_diff', 'over_asking_pct', 'price_per_sqft', 'neighborhood_hotness',
#     'realtor_logic', 'recency_weight', 'market_heat_score',
#     'house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood',
#     'is_hot_area', 'neighborhood_ppsf_premium', 'likely_multi_offer', 'season_boost'
# ]

# df = df[features + [target]].dropna()
# if df.empty:
#     print("[❌] Data is empty after cleaning. Aborting.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# X = df.drop(target, axis=1)
# y = df[target]

# # ========================
# # Train/Test Split
# # ========================
# bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=bins, random_state=42
# )

# categorical = ['house_type', 'garage_type', 'season', 'dom_bucket', 'neighborhood']
# numerical = [col for col in X.columns if col not in categorical]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
# ], remainder='passthrough')

# # ========================
# # Model
# # ========================
# model = LGBMRegressor(
#     n_estimators=3000,
#     learning_rate=0.015,
#     max_depth=10,
#     subsample=0.9,
#     colsample_bytree=0.8,
#     min_child_samples=20,
#     random_state=42,
#     n_jobs=-1
# )

# pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", model)
# ])

# # ========================
# # Fit
# # ========================
# pipeline.named_steps['model'].fit(
#     preprocessor.fit_transform(X_train),
#     y_train,
#     eval_set=[(preprocessor.transform(X_test), y_test)],
#     eval_metric='mae',
#     callbacks=[
#         early_stopping(stopping_rounds=100),
#         log_evaluation(period=50)
#     ]
# )

# y_pred = pipeline.named_steps['model'].predict(preprocessor.transform(X_test))

# mae = mean_absolute_error(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"[✅] Model Trained | MAE=${mae:,.0f} | MAPE={mape:.2%} | R²={r2:.4f}")

# # ========================
# # Save Model
# # ========================
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_path = f"trained_price_model_{timestamp}.pkl"
# joblib.dump(pipeline, model_path)
# joblib.dump(pipeline, "trained_price_model.pkl")
# print(f"✅ Model saved to {model_path} and updated trained_price_model.pkl")

# # file: ml/train_model.py

# import pandas as pd
# import numpy as np
# import joblib
# import ast
# from datetime import datetime
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# print("🚀 Starting model training pipeline with HOT AREAS enhancements...")

# # ========================
# # DB Config and Load
# # ========================
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )

# query = """
# SELECT neighborhood, house_type, bedrooms, bathrooms, sqft,
#        built_year, garage_type, season, sold_price, list_price, listing_date
# FROM housing_data;
# """

# df = pd.read_sql_query(query, engine)

# if df.empty:
#     print("[⚠️] No data found in DB.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# print(f"[ℹ️] Loaded {len(df)} rows from DB")

# # ========================
# # Preprocessing
# # ========================
# df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
# df['age'] = datetime.now().year - df['built_year']

# print(f"[ℹ️] Rows before filters: {len(df)}")
# df = df[(df['bathrooms'] > 0) & (df['bedrooms'] > 0) & (df['sqft'] > 300)]
# print(f"[ℹ️] After bedrooms/bathrooms/sqft filter: {len(df)}")

# df = df.dropna(subset=['sold_price', 'list_price', 'listing_date'])
# print(f"[ℹ️] After dropping NA on sold_price/list_price/listing_date: {len(df)}")

# df = df[(df['sold_price'] > 0) & (df['list_price'] > 0)]
# print(f"[ℹ️] After sold_price and list_price > 0 filter: {len(df)}")

# # Outlier clipping
# df['sold_price'] = df['sold_price'].clip(upper=df['sold_price'].quantile(0.99))
# df['list_price'] = df['list_price'].clip(upper=df['list_price'].quantile(0.99))

# # Market heat features
# recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
# df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)
# df['price_diff'] = df['sold_price'] - df['list_price']
# df['over_asking_pct'] = df['price_diff'] / df['list_price']
# df['price_per_sqft'] = df['sold_price'] / df['sqft']
# df['multi_offer_flag'] = (df['over_asking_pct'] > 0.05).astype(int)
# df['neighborhood_hotness'] = 1 - (
#     df.groupby("neighborhood")['price_per_sqft'].transform("mean") / df['price_per_sqft'].max()
# )

# # Market heat composite
# df['market_heat_score'] = (
#     0.4 * df['recency_weight'] +
#     0.3 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag']
# )

# # Realtor logic feature
# df['realtor_logic'] = (
#     0.3 * (1 - df['age'] / 100) +
#     0.4 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag']
# )

# # ========================
# # Add HOT AREAS Flags
# # ========================
# hot_areas = [
#     "Bridgwater", "Prairie Pointe", "Whyte Ridge", "East Transcona",
#     "River Heights", "St. Vital", "Tuxedo", "North Kildonan",
#     "Osborne Village", "Fort Richmond", "Bridgwater Trails",
#     "Bridgwater Lakes", "Bridgwater Forest", "Bridgwater Centre"
# ]
# df['is_hot_area'] = df['neighborhood'].isin(hot_areas).astype(int)

# # Neighborhood PPSF Premium
# neigh_median_ppsf = df.groupby('neighborhood')['price_per_sqft'].transform('median')
# city_median_ppsf = df['price_per_sqft'].median()
# df['neighborhood_ppsf_premium'] = neigh_median_ppsf / city_median_ppsf

# # Likely multi-offer engineered feature
# df['likely_multi_offer'] = (
#     (df['list_price'] < neigh_median_ppsf * df['sqft'] * 0.98) &
#     (df['market_heat_score'] > 0.6)
# ).astype(int)

# # Season weighted boost
# df['season_boost'] = df['season'].map({
#     'Spring': 1.1,
#     'Summer': 1.05,
#     'Fall': 1.0,
#     'Winter': 0.95
# }).fillna(1.0)

# # ========================
# # Features and Target
# # ========================
# target = "sold_price"
# features = [
#     'bedrooms', 'bathrooms', 'sqft', 'age', 'list_price',
#     'price_diff', 'over_asking_pct', 'price_per_sqft', 'neighborhood_hotness',
#     'realtor_logic', 'recency_weight', 'market_heat_score',
#     'house_type', 'garage_type', 'season', 'neighborhood',
#     'is_hot_area', 'neighborhood_ppsf_premium', 'likely_multi_offer', 'season_boost'
# ]

# df = df[features + [target]].dropna()
# print(f"[ℹ️] Rows before train/test split: {len(df)}")
# if df.empty:
#     print("[❌] Data is empty after cleaning. Aborting.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# X = df.drop(target, axis=1)
# y = df[target]

# # ========================
# # Train/Test Split
# # ========================
# bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=bins, random_state=42
# )

# categorical = ['house_type', 'garage_type', 'season', 'neighborhood']
# numerical = [col for col in X.columns if col not in categorical]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
# ], remainder='passthrough')

# # ========================
# # Model
# # ========================
# model = LGBMRegressor(
#     n_estimators=3000,
#     learning_rate=0.015,
#     max_depth=10,
#     subsample=0.9,
#     colsample_bytree=0.8,
#     min_child_samples=20,
#     random_state=42,
#     n_jobs=-1
# )

# pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", model)
# ])

# # ========================
# # Fit
# # ========================
# pipeline.named_steps['model'].fit(
#     preprocessor.fit_transform(X_train),
#     y_train,
#     eval_set=[(preprocessor.transform(X_test), y_test)],
#     eval_metric='mae',
#     callbacks=[
#         early_stopping(stopping_rounds=100),
#         log_evaluation(period=50)
#     ]
# )

# y_pred = pipeline.named_steps['model'].predict(preprocessor.transform(X_test))

# mae = mean_absolute_error(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"[✅] Model Trained | MAE=${mae:,.0f} | MAPE={mape:.2%} | R²={r2:.4f}")

# # ========================
# # Save Model
# # ========================
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_path = f"trained_price_model_{timestamp}.pkl"
# joblib.dump(pipeline, model_path)
# joblib.dump(pipeline, "trained_price_model.pkl")
# print(f"✅ Model saved to {model_path} and updated trained_price_model.pkl")


# # file: ml/train_model.py

# import pandas as pd
# import numpy as np
# import joblib
# import ast
# from datetime import datetime
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# print("🚀 Starting SamVision AI model training pipeline...")

# # ========================
# # DB Config and Load
# # ========================
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )

# query = """
# SELECT neighborhood, house_type, style, bedrooms, bathrooms, sqft,
#        built_year, garage_type, season, sold_price, list_price, listing_date
# FROM housing_data;
# """

# df = pd.read_sql_query(query, engine)

# if df.empty:
#     print("[⚠️] No data found in DB.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# print(f"[ℹ️] Loaded {len(df)} rows from DB")

# # ========================
# # Preprocessing & Feature Engineering
# # ========================
# df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
# df['age'] = datetime.now().year - df['built_year']

# # Filters
# print(f"[ℹ️] Rows before filters: {len(df)}")
# df = df[
#     (df['bathrooms'] > 0) &
#     (df['bedrooms'] > 0) &
#     (df['sqft'] >= 500) &
#     (df['sold_price'] > 0) &
#     (df['list_price'] > 0)
# ]
# df = df.dropna(subset=['sold_price', 'list_price', 'listing_date'])
# print(f"[ℹ️] Rows after initial filtering: {len(df)}")

# # Outlier clipping
# df['sold_price'] = df['sold_price'].clip(upper=df['sold_price'].quantile(0.99))
# df['list_price'] = df['list_price'].clip(upper=df['list_price'].quantile(0.99))

# # Market heat + Realtor Logic
# recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
# df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)
# df['price_diff'] = df['sold_price'] - df['list_price']
# df['over_asking_pct'] = df['price_diff'] / df['list_price']
# df['price_per_sqft'] = df['sold_price'] / df['sqft']
# df['multi_offer_flag'] = (df['over_asking_pct'] > 0.05).astype(int)
# df['neighborhood_hotness'] = 1 - (
#     df.groupby("neighborhood")['price_per_sqft'].transform("mean") / df['price_per_sqft'].max()
# )
# df['realtor_logic'] = (
#     0.4 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag'] +
#     0.3 * (1 - df['age'] / 100)
# )

# # Hot Areas Flag
# hot_areas = [
#     "Bridgwater", "Prairie Pointe", "Whyte Ridge", "East Transcona",
#     "River Heights", "St. Vital", "Tuxedo", "North Kildonan",
#     "Osborne Village", "Fort Richmond", "Bridgwater Trails",
#     "Bridgwater Lakes", "Bridgwater Forest", "Bridgwater Centre"
# ]
# df['is_hot_area'] = df['neighborhood'].isin(hot_areas).astype(int)

# # Neighborhood PPSF premium
# neigh_median_ppsf = df.groupby('neighborhood')['price_per_sqft'].transform('median')
# city_median_ppsf = df['price_per_sqft'].median()
# df['neighborhood_ppsf_premium'] = neigh_median_ppsf / city_median_ppsf

# # Likely multi-offer flag
# df['likely_multi_offer'] = (
#     (df['list_price'] < neigh_median_ppsf * df['sqft'] * 0.98) &
#     (df['multi_offer_flag'] == 1)
# ).astype(int)

# # Season boost
# df['season_boost'] = df['season'].map({
#     'Spring': 1.1,
#     'Summer': 1.05,
#     'Fall': 1.0,
#     'Winter': 0.95
# }).fillna(1.0)

# # Comps per neighborhood for Realtor fallback alignment
# df['comp_count_in_neighborhood'] = df['neighborhood'].map(df['neighborhood'].value_counts())

# # ========================
# # Features and Target
# # ========================
# target = "sold_price"
# features = [
#     'bedrooms', 'bathrooms', 'sqft', 'age', 'list_price',
#     'price_diff', 'over_asking_pct', 'price_per_sqft',
#     'neighborhood_hotness', 'realtor_logic', 'recency_weight',
#     'multi_offer_flag', 'is_hot_area', 'neighborhood_ppsf_premium',
#     'likely_multi_offer', 'season_boost', 'comp_count_in_neighborhood',
#     'house_type', 'garage_type', 'season', 'neighborhood', 'style'
# ]

# df = df[features + [target]].dropna()
# print(f"[ℹ️] Rows before train/test split: {len(df)}")

# X = df.drop(target, axis=1)
# y = df[target]

# # ========================
# # Train/Test Split
# # ========================
# bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=bins, random_state=42
# )

# categorical = ['house_type', 'garage_type', 'season', 'neighborhood', 'style']
# numerical = [col for col in X.columns if col not in categorical]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
# ], remainder='passthrough')

# # ========================
# # Model
# # ========================
# model = LGBMRegressor(
#     n_estimators=3000,
#     learning_rate=0.015,
#     max_depth=10,
#     subsample=0.9,
#     colsample_bytree=0.8,
#     min_child_samples=20,
#     random_state=42,
#     n_jobs=-1
# )

# pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", model)
# ])

# # ========================
# # Fit Model
# # ========================
# pipeline.named_steps['model'].fit(
#     preprocessor.fit_transform(X_train),
#     y_train,
#     eval_set=[(preprocessor.transform(X_test), y_test)],
#     eval_metric='mae',
#     callbacks=[
#         early_stopping(stopping_rounds=100),
#         log_evaluation(period=50)
#     ]
# )

# y_pred = pipeline.named_steps['model'].predict(preprocessor.transform(X_test))
# mae = mean_absolute_error(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"[✅] Model Trained | MAE=${mae:,.0f} | MAPE={mape:.2%} | R²={r2:.4f}")

# # ========================
# # Save Model
# # ========================
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_path = f"trained_price_model_{timestamp}.pkl"
# joblib.dump(pipeline, model_path)
# joblib.dump(pipeline, "trained_price_model.pkl")
# print(f"✅ Model saved as {model_path} and updated trained_price_model.pkl")

# # file: ml/train_model.py

# import pandas as pd
# import numpy as np
# import joblib
# import ast
# import os
# from datetime import datetime
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# print("🚀 Starting SamVision AI model training pipeline with Realtor Prioritization...")

# # ========================
# # DB Config & Load
# # ========================
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )

# query = """
# SELECT neighborhood, house_type, style, bedrooms, bathrooms, sqft,
#        built_year, garage_type, season, sold_price, list_price, listing_date
# FROM housing_data;
# """

# df = pd.read_sql_query(query, engine)

# if df.empty:
#     print("[⚠️] No data found in DB. Exiting.")
#     joblib.dump(None, 'trained_price_model.pkl')
#     exit()

# print(f"[ℹ️] Loaded {len(df)} rows from DB")

# # ========================
# # Preprocessing & Feature Engineering
# # ========================
# df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')
# df['age'] = datetime.now().year - df['built_year']

# print(f"[ℹ️] Rows before filters: {len(df)}")
# df = df[
#     (df['bathrooms'] > 0) &
#     (df['bedrooms'] > 0) &
#     (df['sqft'] >= 500) &
#     (df['sold_price'] > 0) &
#     (df['list_price'] > 0)
# ].dropna(subset=['sold_price', 'list_price', 'listing_date'])
# print(f"[ℹ️] Rows after filtering: {len(df)}")

# # Outlier capping
# df['sold_price'] = df['sold_price'].clip(upper=df['sold_price'].quantile(0.99))
# df['list_price'] = df['list_price'].clip(upper=df['list_price'].quantile(0.99))

# # Feature engineering
# recent_cutoff = df['listing_date'].max() - pd.Timedelta(days=90)
# df['recency_weight'] = (df['listing_date'] >= recent_cutoff).astype(int)
# df['price_diff'] = df['sold_price'] - df['list_price']
# df['over_asking_pct'] = df['price_diff'] / df['list_price']
# df['price_per_sqft'] = df['sold_price'] / df['sqft']
# df['multi_offer_flag'] = (df['over_asking_pct'] > 0.05).astype(int)
# df['neighborhood_hotness'] = 1 - (
#     df.groupby('neighborhood')['price_per_sqft'].transform('mean') / df['price_per_sqft'].max()
# )
# df['realtor_logic'] = (
#     0.4 * df['neighborhood_hotness'] +
#     0.3 * df['multi_offer_flag'] +
#     0.3 * (1 - df['age'] / 100)
# )

# hot_areas = [
#     "Bridgwater", "Prairie Pointe", "Whyte Ridge", "East Transcona", "River Heights",
#     "St. Vital", "Tuxedo", "North Kildonan", "Osborne Village", "Fort Richmond",
#     "Bridgwater Trails", "Bridgwater Lakes", "Bridgwater Forest", "Bridgwater Centre"
# ]
# df['is_hot_area'] = df['neighborhood'].isin(hot_areas).astype(int)

# neigh_median_ppsf = df.groupby('neighborhood')['price_per_sqft'].transform('median')
# city_median_ppsf = df['price_per_sqft'].median()
# df['neighborhood_ppsf_premium'] = neigh_median_ppsf / city_median_ppsf

# df['likely_multi_offer'] = (
#     (df['list_price'] < neigh_median_ppsf * df['sqft'] * 0.98) &
#     (df['multi_offer_flag'] == 1)
# ).astype(int)

# df['season_boost'] = df['season'].map({
#     'Spring': 1.1, 'Summer': 1.05, 'Fall': 1.0, 'Winter': 0.95
# }).fillna(1.0)

# df['comp_count_in_neighborhood'] = df['neighborhood'].map(df['neighborhood'].value_counts())

# # ========================
# # Features & Target
# # ========================
# target = "sold_price"
# features = [
#     'bedrooms', 'bathrooms', 'sqft', 'age', 'list_price',
#     'price_diff', 'over_asking_pct', 'price_per_sqft',
#     'neighborhood_hotness', 'realtor_logic', 'recency_weight',
#     'multi_offer_flag', 'is_hot_area', 'neighborhood_ppsf_premium',
#     'likely_multi_offer', 'season_boost', 'comp_count_in_neighborhood',
#     'house_type', 'garage_type', 'season', 'neighborhood', 'style'
# ]

# df = df[features + [target]].dropna()
# print(f"[ℹ️] Final training set size: {len(df)} rows")

# X = df.drop(columns=[target])
# y = df[target]

# # ========================
# # Train/Test Split
# # ========================
# bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=bins, random_state=42
# )

# categorical = ['house_type', 'garage_type', 'season', 'neighborhood', 'style']
# numerical = [col for col in X.columns if col not in categorical]

# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
# ], remainder='passthrough')

# # ========================
# # Model
# # ========================
# model = LGBMRegressor(
#     n_estimators=3000,
#     learning_rate=0.015,
#     max_depth=10,
#     subsample=0.9,
#     colsample_bytree=0.8,
#     min_child_samples=20,
#     random_state=42,
#     n_jobs=-1
# )

# pipeline = Pipeline([
#     ("prep", preprocessor),
#     ("model", model)
# ])

# # ========================
# # Training
# # ========================
# pipeline.named_steps['model'].fit(
#     preprocessor.fit_transform(X_train),
#     y_train,
#     eval_set=[(preprocessor.transform(X_test), y_test)],
#     eval_metric='mae',
#     callbacks=[
#         early_stopping(stopping_rounds=100),
#         log_evaluation(period=50)
#     ]
# )

# y_pred = pipeline.named_steps['model'].predict(preprocessor.transform(X_test))
# mae = mean_absolute_error(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"[✅] Model trained successfully | MAE=${mae:,.0f} | MAPE={mape:.2%} | R²={r2:.4f}")

# # ========================
# # Save Model & Log
# # ========================

# # ========================
# # Save Model Explanation JSON
# # ========================
# import json

# explanation_payload = {
#     "model_used": "LightGBMRegressor with Realtor Intelligence Priority",
#     "trained_on_rows": len(df),
#     "mae": float(mae),
#     "mape": float(mape),
#     "r2": float(r2),
#     "features_used": features,
#     "note": (
#         "This model uses Realtor Intelligence when >=6 comps in the same neighborhood "
#         "in the last 90 days within ±300 sqft, ±1 bed/bath, ±7 years built for direct estimation, "
#         "otherwise falls back to ML model for price prediction."
#     )
# }

# with open("model_explanation.json", "w") as f:
#     json.dump(explanation_payload, f, indent=4)

# print("[💡] model_explanation.json saved for Streamlit UI explainability.")
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# model_filename = f"trained_price_model_{timestamp}.pkl"
# joblib.dump(pipeline, model_filename)
# joblib.dump(pipeline, "trained_price_model.pkl")
# print(f"[💾] Model saved as {model_filename} and updated trained_price_model.pkl")

# # Log training summary for your LLM & SaaS dashboard explainability
# log_entry = {
#     "timestamp": timestamp,
#     "model_path": model_filename,
#     "mae": mae,
#     "mape": mape,
#     "r2": r2,
#     "rows_trained_on": len(df),
#     "features": features
# }

# pd.DataFrame([log_entry]).to_csv("model_training_logs.csv", mode='a', header=not os.path.exists("model_training_logs.csv"), index=False)
# print(explanation_payload)

# print("[🗂️] Training log updated in model_training_logs.csv")

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

print("🚀 Starting SamVision AI model training pipeline with Realtor Multi-Offer Logic...")

# ========================
# DB Config & Load
# ========================
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

engine = create_engine(
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
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

