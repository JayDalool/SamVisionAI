# # # file: app/streamlit_app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import sys
# from datetime import datetime
# import ast
# from sqlalchemy import create_engine

# # Fix module path to access utils
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# st.set_page_config(page_title="SamVision AI", layout="wide")

# st.title("🏡 SamVision AI - Real Estate Analyzer")

# pdf_dir = "pdf_uploads"
# parsed_csv_path = "parsed_csv/merged.csv"
# model_path = "trained_price_model.pkl"
# model = joblib.load(model_path) if os.path.exists(model_path) else None

# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )



# def calculate_features(df):
#     df['price_per_sqft'] = (df['sold_price'] / df['sqft']).round(2)
#     df['age'] = datetime.now().year - df.get('built_year', datetime.now().year)
#     return df

# def insert_sales_to_db(df, db_config):
#     df.to_sql('housing_data', con=engine, if_exists='append', index=False)

# def clean_sales_data(df):
#     df = df.drop_duplicates(subset=['mls_number', 'address'], keep='last')
#     df = df[df['list_price'] > 0]
#     return df

# def load_data():
#     return pd.read_sql_query("SELECT * FROM housing_data", engine)

# df_all = load_data()

# tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 CMA Tool", "💰 Price Prediction"])

# with tab1:
#     st.subheader("📥 Upload & Parse PDF Listings")
#     uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
#     if uploaded_files:
#         all_data = []
#         for uploaded_file in uploaded_files:
#             save_path = os.path.join(pdf_dir, uploaded_file.name)
#             with open(save_path, "wb") as f:
#                 f.write(uploaded_file.read())
#             st.write(f"Parsing: {uploaded_file.name}")
#             try:
#                 df = extract_pdf_sales(save_path)
#                 if not df.empty:
#                     all_data.append(df)
#             except Exception as e:
#                 st.error(f"Failed to parse {uploaded_file.name}: {e}")

#         if all_data and st.button("📤 Load Parsed Data to DB"):
#             parsed_df = pd.concat(all_data, ignore_index=True)
#             parsed_df = clean_sales_data(parsed_df)
#             insert_sales_to_db(parsed_df, db_config)
#             df_all = load_data()
#             st.success("Data loaded into database.")

#     st.header("📊 Current Housing Dataset")
#     if not df_all.empty:
#         st.dataframe(df_all)
#     else:
#         st.info("No data available yet. Upload PDFs to begin.")

# with tab2:
#     st.header("📊 Comparative Market Analysis")
#     if df_all.empty:
#         st.warning("No data available. Please upload and load PDF sales.")
#     else:
#         df_all['Select'] = False
#         selected_rows = st.data_editor(
#             df_all[['Select', 'neighborhood', 'house_type', 'garage_type', 'sqft',
#                     'bedrooms', 'bathrooms', 'sold_price']],
#             use_container_width=True, key="cma_editor"
#         )
#         selected_cma = selected_rows[selected_rows['Select']]
#         if not selected_cma.empty:
#             selected_cma['price_per_sqft'] = (selected_cma['sold_price'] / selected_cma['sqft']).round(2)
#             st.dataframe(selected_cma)
#             st.download_button("📥 Download CMA", selected_cma.to_csv(index=False), "cma_report.csv", "text/csv")

# with tab3:
#     st.header("💰 Predict House Price")
#     if df_all.empty or model is None:
#         st.error("No data available or model missing. Upload and parse first.")
#     else:
#         default = df_all.iloc[0]
#         default_sqft = int(default.get('sqft', 1200))
#         default_sqft = default_sqft if default_sqft >= 600 else 600

#         col1, col2, col3 = st.columns(3)
#         with col1:
#             address = st.text_input("Address", default['address'])
#             neighborhood = st.selectbox("Neighborhood", sorted(df_all['neighborhood'].dropna().unique()))
#             house_type = st.selectbox("House Type", sorted(df_all['house_type'].dropna().unique()))

#         with col2:
#             bedrooms = st.slider("Bedrooms", 1, 6, int(default['bedrooms']))
#             bathrooms = st.slider("Bathrooms", 1, 4, int(default['bathrooms']))
#             sqft = st.number_input("Sqft", min_value=600, max_value=5000, value=default_sqft)

#         with col3:
#             built_year = st.slider("Built Year", 1900, datetime.now().year, int(default.get('built_year', 2000)))
#             garage_type = st.selectbox("Garage Type", sorted(df_all['garage_type'].dropna().unique()))
#             season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

#         dom = st.slider("🏷️ Days on Market (DOM)", 0, 180, int(default.get('dom', 15)))
#         age = datetime.now().year - built_year
#         list_price = st.number_input("List Price", min_value=0, value=int(default.get('list_price', 0)))
#         price_diff = 0
#         over_asking_pct = 0
#         price_per_sqft = list_price / max(sqft, 1)
#         dom_bucket = pd.cut(pd.Series([dom]), bins=[-1, 7, 14, 30, 90, 180], labels=['0-7', '8-14', '15-30', '31-90', '90+'])[0]

#         if st.button("🔮 Predict Price"):
#             df_input = pd.DataFrame([{
#                 'bedrooms': bedrooms,
#                 'bathrooms': bathrooms,
#                 'sqft': sqft,
#                 'age': age,
#                 'dom': dom,
#                 'list_price': list_price,
#                 'price_diff': price_diff,
#                 'over_asking_pct': over_asking_pct,
#                 'price_per_sqft': price_per_sqft,
#                 'neighborhood_hotness': 0.5,
#                 'house_type': house_type,
#                 'garage_type': garage_type,
#                 'season': season,
#                 'dom_bucket': dom_bucket,
#                 'neighborhood': neighborhood
#             }])

#             df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
#             predicted_total = model.predict(df_input)[0]
#             predicted_overpay = predicted_total - list_price
#             predicted_price = list_price + predicted_overpay
#             buffer = max(5000, min(abs(predicted_overpay * 0.5), 10000))

#             st.markdown(f"<h2 style='color:green;'>🏆 Winning Offer Price: ${predicted_price:,.0f}</h2>", unsafe_allow_html=True)
#             st.caption(f"💡 Offer ${predicted_overpay:,.0f} over asking. Confidence ±${buffer:,.0f}")
# file: app/streamlit_app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import sys
# from datetime import datetime
# import ast
# from sqlalchemy import create_engine

# # Fix module path to access utils
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# st.set_page_config(page_title="SamVision AI", layout="wide")

# st.title("🏡 SamVision AI - Real Estate Analyzer")

# pdf_dir = "pdf_uploads"
# parsed_csv_path = "parsed_csv/merged.csv"
# model_path = "trained_price_model.pkl"
# model = joblib.load(model_path) if os.path.exists(model_path) else None

# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )


# def calculate_features(df):
#     df['price_per_sqft'] = (df['sold_price'] / df['sqft']).round(2)
#     df['age'] = datetime.now().year - df.get('built_year', datetime.now().year)
#     return df

# def insert_sales_to_db(df, db_config):
#     df.to_sql('housing_data', con=engine, if_exists='append', index=False)

# def clean_sales_data(df):
#     df = df.drop_duplicates(subset=['mls_number', 'address'], keep='last')
#     df = df[df['list_price'] > 0]
#     df['neighborhood'] = df['neighborhood'].str.strip().str.title()
#     return df

# def load_data():
#     return pd.read_sql_query("SELECT * FROM housing_data", engine)

# df_all = load_data()

# tab1, tab2, tab3 = st.tabs(["🏡 Home", "📊 CMA Tool", "💰 Price Prediction"])

# with tab1:
#     st.subheader("📥 Upload & Parse PDF Listings")
#     uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
#     if uploaded_files:
#         all_data = []
#         for uploaded_file in uploaded_files:
#             save_path = os.path.join(pdf_dir, uploaded_file.name)
#             with open(save_path, "wb") as f:
#                 f.write(uploaded_file.read())
#             st.write(f"Parsing: {uploaded_file.name}")
#             try:
#                 df = extract_pdf_sales(save_path)
#                 if not df.empty:
#                     all_data.append(df)
#             except Exception as e:
#                 st.error(f"Failed to parse {uploaded_file.name}: {e}")

#         if all_data and st.button("📤 Load Parsed Data to DB"):
#             parsed_df = pd.concat(all_data, ignore_index=True)
#             parsed_df = clean_sales_data(parsed_df)
#             insert_sales_to_db(parsed_df, db_config)
#             df_all = load_data()
#             st.success("Data loaded into database.")

#     st.header("📊 Current Housing Dataset")
#     if not df_all.empty:
#         st.dataframe(df_all)
#     else:
#         st.info("No data available yet. Upload PDFs to begin.")

# with tab2:
#     st.header("📊 Comparative Market Analysis")
#     if df_all.empty:
#         st.warning("No data available. Please upload and load PDF sales.")
#     else:
#         df_all['Select'] = False
#         selected_rows = st.data_editor(
#             df_all[['Select', 'neighborhood', 'house_type', 'garage_type', 'sqft',
#                     'bedrooms', 'bathrooms', 'sold_price']],
#             use_container_width=True, key="cma_editor"
#         )
#         selected_cma = selected_rows[selected_rows['Select']]
#         if not selected_cma.empty:
#             selected_cma['price_per_sqft'] = (selected_cma['sold_price'] / selected_cma['sqft']).round(2)
#             st.dataframe(selected_cma)
#             st.download_button("📥 Download CMA", selected_cma.to_csv(index=False), "cma_report.csv", "text/csv")

# with tab3:
#     st.header("💰 Predict House Price")
# if df_all.empty or model is None:
#     st.error("No data available or model missing. Upload and parse first.")
# else:
#     default = df_all.iloc[0]
#     default_sqft = int(default.get('sqft', 1200))
#     default_sqft = default_sqft if default_sqft >= 600 else 600

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         bedrooms = st.slider("Bedrooms", 1, 6, int(default['bedrooms']))
#         bathrooms = st.slider("Bathrooms", 1, 4, int(default['bathrooms']))
#         sqft = st.number_input("Sqft", min_value=600, max_value=5000, value=default_sqft)

#     with col2:
#         built_year = st.slider("Built Year", 1900, datetime.now().year, int(default.get('built_year', 2000)))
#         age = datetime.now().year - built_year
#         dom = st.slider("🏷️ Days on Market (DOM)", 0, 180, int(default.get('dom', 15)))

#     with col3:
#         list_price = st.number_input("List Price", min_value=0, value=int(default.get('list_price', 0)))
#         house_type = st.selectbox("House Type", sorted(df_all['house_type'].dropna().unique()))
#         garage_type = st.selectbox("Garage Type", sorted(df_all['garage_type'].dropna().unique()))

#     season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])
#     neighborhood = st.selectbox("Neighborhood", sorted(df_all['neighborhood'].dropna().unique()))

#     if st.button("🔮 Predict Price"):
#         df_input = pd.DataFrame([{
#             'bedrooms': bedrooms,
#             'bathrooms': bathrooms,
#             'sqft': sqft,
#             'age': age,
#             'dom': dom,
#             'list_price': list_price,
#             'house_type': house_type,
#             'garage_type': garage_type,
#             'season': season,
#             'neighborhood': neighborhood
#         }])

#         df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
#         predicted_price = model.predict(df_input)[0]
#         buffer = max(5000, min(abs(predicted_price * 0.05), 10000))

#         st.markdown(f"<h2 style='color:green;'>🏆 Predicted Price: ${predicted_price:,.0f}</h2>", unsafe_allow_html=True)
#         st.caption(f"Confidence Interval ±${buffer:,.0f} → ${predicted_price - buffer:,.0f} - ${predicted_price + buffer:,.0f}")
# file: app/streamlit_app.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import sys
# from datetime import datetime
# import ast
# from sqlalchemy import create_engine

# # Fix module path to access utils
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from utils.pdf_sales_parser import extract_pdf_sales  # ensure imported

# st.set_page_config(page_title="SamVision AI", layout="wide")

# st.title("🏡 SamVision AI - Real Estate Analyzer")

# pdf_dir = "pdf_uploads"
# parsed_csv_path = "parsed_csv/merged.csv"
# model_path = "trained_price_model.pkl"
# model = joblib.load(model_path) if os.path.exists(model_path) else None

# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )

# def calculate_features(df):
#     df['price_per_sqft'] = (df['sold_price'] / df['sqft']).round(2)
#     df['age'] = datetime.now().year - df.get('built_year', datetime.now().year)
#     return df

# def insert_sales_to_db(df, db_config):
#     df.to_sql('housing_data', con=engine, if_exists='append', index=False)

# def clean_sales_data(df):
#     df = df.drop_duplicates(subset=['mls_number', 'address'], keep='last')
#     df = df[df['list_price'] > 0]
#     return df

# def load_data():
#     return pd.read_sql_query("SELECT * FROM housing_data", engine)

# df_all = load_data()

# tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 CMA Tool", "💰 Price Prediction"])

# with tab1:
#     st.subheader("📥 Upload & Parse PDF Listings")
#     uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
#     if uploaded_files:
#         all_data = []
#         for uploaded_file in uploaded_files:
#             save_path = os.path.join(pdf_dir, uploaded_file.name)
#             with open(save_path, "wb") as f:
#                 f.write(uploaded_file.read())
#             st.write(f"Parsing: {uploaded_file.name}")
#             try:
#                 df = extract_pdf_sales(save_path)
#                 if not df.empty:
#                     all_data.append(df)
#             except Exception as e:
#                 st.error(f"Failed to parse {uploaded_file.name}: {e}")

#         if all_data and st.button("📤 Load Parsed Data to DB"):
#             parsed_df = pd.concat(all_data, ignore_index=True)
#             parsed_df = clean_sales_data(parsed_df)
#             insert_sales_to_db(parsed_df, db_config)
#             df_all = load_data()
#             st.success("Data loaded into database.")

#     st.header("📊 Current Housing Dataset")
#     if not df_all.empty:
#         st.dataframe(df_all)
#     else:
#         st.info("No data available yet. Upload PDFs to begin.")

# with tab2:
#     st.header("📊 Comparative Market Analysis")
#     if df_all.empty:
#         st.warning("No data available. Please upload and load PDF sales.")
#     else:
#         df_all['Select'] = False
#         selected_rows = st.data_editor(
#             df_all[['Select', 'neighborhood', 'house_type', 'garage_type', 'sqft',
#                     'bedrooms', 'bathrooms', 'sold_price']],
#             use_container_width=True, key="cma_editor"
#         )
#         selected_cma = selected_rows[selected_rows['Select']]
#         if not selected_cma.empty:
#             selected_cma['price_per_sqft'] = (selected_cma['sold_price'] / selected_cma['sqft']).round(2)
#             st.dataframe(selected_cma)
#             st.download_button("📥 Download CMA", selected_cma.to_csv(index=False), "cma_report.csv", "text/csv")

# with tab3:
#     st.header("💰 Predict House Price")
#     if df_all.empty or model is None:
#         st.error("No data available or model missing. Upload and parse first.")
#     else:
#         default = df_all.iloc[0]
#         default_sqft = int(default.get('sqft', 1200))
#         default_sqft = default_sqft if default_sqft >= 600 else 600

#         col1, col2, col3 = st.columns(3)
#         with col1:
#             address = st.text_input("Address", default['address'])
#             neighborhood = st.selectbox("Neighborhood", sorted(df_all['neighborhood'].dropna().unique()))
#             house_type = st.selectbox("House Type", sorted(df_all['house_type'].dropna().unique()))

#         with col2:
#             bedrooms = st.text_input("Bedrooms", str(int(default.get('bedrooms', 3))))
#             bathrooms = st.text_input("Bathrooms", str(float(default.get('bathrooms', 2))))
#             sqft = st.text_input("Sqft", str(default_sqft))

#         with col3:
#             built_year = st.text_input("Built Year", str(int(default.get('built_year', 2000))))
#             garage_type = st.selectbox("Garage Type", sorted(df_all['garage_type'].dropna().unique()))
#             season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

#         dom = st.text_input("🏷️ Days on Market (DOM)", str(int(default.get('dom', 15))))
#         list_price = st.text_input("List Price", str(int(default.get('list_price', 300000))))

#         # Conversion handling:
#         try:
#             bedrooms_val = int(bedrooms)
#             bathrooms_val = float(bathrooms)
#             sqft_val = int(sqft)
#             built_year_val = int(built_year)
#             dom_val = int(dom)
#             list_price_val = int(list_price)
#         except:
#             st.error("Please ensure all fields are entered as numbers where appropriate.")
#             st.stop()

#         age = datetime.now().year - built_year_val
#         price_diff = 0
#         over_asking_pct = 0
#         price_per_sqft = list_price_val / max(sqft_val, 1)
#         dom_bucket = pd.cut(pd.Series([dom_val]), bins=[-1, 7, 14, 30, 90, 180],
#                             labels=['0-7', '8-14', '15-30', '31-90', '90+'])[0]

#         if st.button("🔮 Predict Price"):
#             df_input = pd.DataFrame([{
#                 'bedrooms': bedrooms_val,
#                 'bathrooms': bathrooms_val,
#                 'sqft': sqft_val,
#                 'age': age,
#                 'dom': dom_val,
#                 'list_price': list_price_val,
#                 'price_diff': price_diff,
#                 'over_asking_pct': over_asking_pct,
#                 'price_per_sqft': price_per_sqft,
#                 'neighborhood_hotness': 0.5,
#                 'house_type': house_type,
#                 'garage_type': garage_type,
#                 'season': season,
#                 'dom_bucket': dom_bucket,
#                 'neighborhood': neighborhood
#             }])

#             df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
#             predicted_total = model.predict(df_input)[0]
#             predicted_overpay = predicted_total - list_price_val
#             predicted_price = list_price_val + predicted_overpay
#             buffer = max(5000, min(abs(predicted_overpay * 0.5), 10000))

#             st.markdown(f"<h2 style='color:green;'>🏆 Winning Offer Price: ${predicted_price:,.0f}</h2>", unsafe_allow_html=True)
#             st.caption(f"💡 Offer ${predicted_overpay:,.0f} over asking. Confidence ±${buffer:,.0f}")
# import streamlit as st

# # Must be the first Streamlit command
# st.set_page_config(page_title="🏡 SamVision AI - Realtor CMA + Multi-Offer Predictor", layout="wide")

# # --- Core Imports ---
# import pandas as pd
# import numpy as np
# import os
# import ast
# import joblib
# import sys
# from datetime import datetime
# from sqlalchemy import create_engine
# import plotly.express as px


# # Ensure utils can be imported cleanly
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.pdf_sales_parser import extract_pdf_sales

# # --- CONFIG ---
# pdf_dir = "pdf_uploads"
# model_path = "trained_price_model.pkl"
# os.makedirs(pdf_dir, exist_ok=True)

# # Load trained model
# model = joblib.load(model_path) if os.path.exists(model_path) else None

# # Load DB config
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# engine = create_engine(
#     f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
# )

# # --- UTILS ---
# def insert_sales_to_db(df):
#     df.to_sql('housing_data', con=engine, if_exists='append', index=False)

# def clean_sales_data(df):
#     df = df.drop_duplicates(subset=['mls_number', 'address'], keep='last')
#     df = df[df['list_price'] > 0]
#     return df

# @st.cache_resource(show_spinner=False)
# def load_data():
#     try:
#         return pd.read_sql_query("SELECT * FROM housing_data", engine)
#     except Exception as e:
#         st.error(f"DB load failed: {e}")
#         return pd.DataFrame()

# # --- HEADER ---
# st.title("🏡 SamVision AI - Realtor CMA + Multi-Offer Price Predictor")

# st.markdown("""
# **Workflow for Realtors:**
# 1️⃣ **Upload MLS PDFs to auto-extract sold listings**  
# 2️⃣ **Load parsed data into your database** for analysis  
# 3️⃣ Generate **Comparative Market Analyses (CMA) easily**  
# 4️⃣ Predict **winning offer prices** within ±$10,000 for competitive listings
# """)

# df_all = load_data()

# # --- TABS ---
# tab1, tab2, tab3 = st.tabs(["📥 Upload & Parse", "📊 CMA Tool", "💰 Price Prediction"])

# # --- TAB 1: Upload & Parse ---
# with tab1:
#     st.header("📥 Upload & Parse MLS PDFs")
#     uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

#     if uploaded_files:
#         all_data = []
#         with st.spinner("Parsing PDFs..."):
#             for uploaded_file in uploaded_files:
#                 save_path = os.path.join(pdf_dir, uploaded_file.name)
#                 with open(save_path, "wb") as f:
#                     f.write(uploaded_file.read())

#                 try:
#                     df = extract_pdf_sales(save_path)
#                     if not df.empty:
#                         all_data.append(df)
#                         st.success(f"✅ Parsed {len(df)} rows from {uploaded_file.name}")
#                     else:
#                         st.warning(f"⚠️ No data extracted from {uploaded_file.name}")
#                 except Exception as e:
#                     st.error(f"❌ Failed to parse {uploaded_file.name}: {e}")
#                 finally:
#                     os.remove(save_path)

#         if all_data:
#             parsed_df = pd.concat(all_data, ignore_index=True)
#             parsed_df = clean_sales_data(parsed_df)

#             if st.button("📤 Load Parsed Data to Database"):
#                 try:
#                     with st.spinner("Loading data into database..."):
#                         insert_sales_to_db(parsed_df)
#                     st.success(f"✅ Loaded {len(parsed_df)} rows into the database.")
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"❌ Failed to load data: {e}")

#     st.subheader("📊 Current Housing Dataset")
#     if not df_all.empty:
#         st.dataframe(df_all, use_container_width=True)
#     else:
#         st.info("No data available yet. Upload PDFs to get started.")

# # --- TAB 2: CMA TOOL ---
# with tab2:
#     st.header("📊 Comparative Market Analysis Tool")
#     if df_all.empty:
#         st.warning("Please upload and load data first.")
#     else:
#         df_all['Select'] = False
#         selected_rows = st.data_editor(
#             df_all[['Select', 'neighborhood', 'house_type', 'garage_type', 'sqft',
#                     'bedrooms', 'bathrooms', 'sold_price']],
#             use_container_width=True,
#             key="cma_editor"
#         )
#         selected_cma = selected_rows[selected_rows['Select']]

#         if not selected_cma.empty:
#             selected_cma['price_per_sqft'] = (selected_cma['sold_price'] / selected_cma['sqft']).round(2)
#             st.dataframe(selected_cma, use_container_width=True)
#             st.download_button(
#                 "📥 Download CMA Report",
#                 selected_cma.to_csv(index=False),
#                 "cma_report.csv",
#                 "text/csv"
#             )
#         else:
#             st.info("Select listings above to generate a CMA report.")

# # --- TAB 3: PRICE PREDICTION ---
# with tab3:
#     st.header("💰 Predict Winning Offer Price")
#     if df_all.empty or model is None:
#         st.error("No data or trained model available. Upload and parse data, then train your model first.")
#     else:
#         default = df_all.iloc[0]
#         col1, col2, col3 = st.columns(3)

#         with col1:
#             neighborhood = st.selectbox("Neighborhood", sorted(df_all['neighborhood'].dropna().unique()), index=0)
#             house_type = st.selectbox("House Type", sorted(df_all['house_type'].dropna().unique()), index=0)
#             garage_type = st.selectbox("Garage Type", sorted(df_all['garage_type'].dropna().unique()), index=0)

#         with col2:
#             bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=int(default.get('bedrooms', 3)))
#             bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=6.0, value=float(default.get('bathrooms', 2.0)))
#             sqft = st.number_input("Square Footage", min_value=300, max_value=6000, value=int(default.get('sqft', 1200)))

#         with col3:
#             default_built_year = int(default.get('built_year', 2000))
#             if default_built_year < 1900 or default_built_year > datetime.now().year:
#                 default_built_year = 2000

#             built_year = st.number_input(
#                 "Built Year",
#                 min_value=1900,
#                 max_value=datetime.now().year,
#                 value=default_built_year
#             )

#             season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'], index=2)
#             dom = st.number_input("Days on Market (DOM)", min_value=0, max_value=365, value=int(default.get('dom', 14)))
#             list_price = st.number_input("List Price", min_value=50000, max_value=2000000, value=int(default.get('list_price', 300000)))

#         age = datetime.now().year - built_year
#         price_diff = 0
#         over_asking_pct = 0
#         price_per_sqft = list_price / max(sqft, 1)
#         dom_bucket = pd.cut(pd.Series([dom]), bins=[-1, 7, 14, 30, 90, 180],
#                             labels=['0-7', '8-14', '15-30', '31-90', '90+'])[0]

#         if st.button("🔮 Predict Winning Offer Price"):
#             try:
#                 input_df = pd.DataFrame([{
#                     'bedrooms': bedrooms,
#                     'bathrooms': bathrooms,
#                     'sqft': sqft,
#                     'age': age,
#                     'dom': dom,
#                     'list_price': list_price,
#                     'price_diff': price_diff,
#                     'over_asking_pct': over_asking_pct,
#                     'price_per_sqft': price_per_sqft,
#                     'neighborhood_hotness': 0.5,
#                     'realtor_logic': 0.5,
#                     'recency_weight': 1,
#                     'house_type': house_type,
#                     'garage_type': garage_type,
#                     'season': season,
#                     'dom_bucket': dom_bucket,
#                     'neighborhood': neighborhood
#                 }])

#                 input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
#                 predicted_price = model.predict(input_df)[0]
#                 buffer = max(5000, min(abs(predicted_price * 0.02), 10000))

#                 st.session_state["predicted_price"] = predicted_price
#                 st.session_state["buffer"] = buffer

#             except Exception as e:
#                 st.error(f"Prediction failed: {e}")

#         # Display predicted price if exists
#         if "predicted_price" in st.session_state:
#             predicted_price = st.session_state["predicted_price"]
#             buffer = st.session_state["buffer"]
#             st.success(f"🏆 Recommended Winning Offer: ${predicted_price:,.0f} ±${buffer:,.0f}")

#             if st.toggle("Show Feature Importance"):
#                 try:
#                     model_obj = model.named_steps['model'] if 'model' in model.named_steps else model

#                     if hasattr(model_obj, 'feature_importances_'):
#                         st.subheader("🔍 Top Predictive Features")

#                         # Extract expanded feature names from the preprocessor
#                         feature_names = model.named_steps['prep'].get_feature_names_out()

#                         # Build series for plotting
#                         feature_imp_series = pd.Series(
#                             model_obj.feature_importances_,
#                             index=feature_names
#                         ).sort_values(ascending=False).head(15)


#                         # Take top 15 by absolute importance
#                         feature_imp_df = feature_imp_series.abs().sort_values(ascending=False).head(15).reset_index()
#                         feature_imp_df.columns = ['Feature', 'Importance']

#                         fig = px.bar(
#                             feature_imp_df,
#                             x='Importance',
#                             y='Feature',
#                             orientation='h',
#                             title="🔍 Top 15 Predictive Features",
#                             color='Importance',
#                             color_continuous_scale='Blues',
#                             height=600
#                         )

#                         fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)

#                         st.plotly_chart(fig, use_container_width=True)

#                     else:
#                         st.info("Feature importance not available for this model.")
#                 except Exception as e:
#                     st.error(f"Error displaying feature importance: {e}")

import streamlit as st

# MUST be first
st.set_page_config(page_title="🏡 SamVision AI - Realtor CMA + Multi-Offer Predictor", layout="wide")

# --- Core Imports ---
import pandas as pd
import numpy as np
import os
import ast
import joblib
import sys
from datetime import datetime
from sqlalchemy import create_engine
import plotly.express as px

# Ensure utils import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.pdf_sales_parser import extract_pdf_sales

# --- CONFIG ---
pdf_dir = "pdf_uploads"
model_path = "trained_price_model.pkl"
os.makedirs(pdf_dir, exist_ok=True)

# Load trained model
model = joblib.load(model_path) if os.path.exists(model_path) else None

# Load DB config
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

engine = create_engine(
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)

# --- UTILS ---
def insert_sales_to_db(df):
    df.to_sql('housing_data', con=engine, if_exists='append', index=False)

def clean_sales_data(df):
    df = df.drop_duplicates(subset=['mls_number', 'address'], keep='last')
    df = df[df['list_price'] > 0]
    return df

@st.cache_resource(show_spinner=False)
def load_data():
    try:
        return pd.read_sql_query("SELECT * FROM housing_data", engine)
    except Exception as e:
        st.error(f"DB load failed: {e}")
        return pd.DataFrame()

# --- HEADER ---
st.title("🏡 SamVision AI - Realtor CMA + Multi-Offer Price Predictor")
st.markdown("""
**Workflow for Realtors:**
1️⃣ **Upload MLS PDFs to auto-extract sold listings**  
2️⃣ **Load parsed data into your database** for analysis  
3️⃣ Generate **Comparative Market Analyses (CMA) easily**  
4️⃣ Predict **winning offer prices** within ±$10,000 for competitive listings
""")

df_all = load_data()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📥 Upload & Parse", "📊 CMA Tool", "💰 Price Prediction"])

# =============================
# 📥 TAB 1: Upload & Parse PDFs
# =============================
with tab1:
    st.header("📥 Upload & Parse MLS PDFs")
    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        all_data = []
        with st.spinner("Parsing PDFs..."):
            for uploaded_file in uploaded_files:
                save_path = os.path.join(pdf_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.read())

                try:
                    df = extract_pdf_sales(save_path)
                    if not df.empty:
                        all_data.append(df)
                        st.success(f"✅ Parsed {len(df)} rows from {uploaded_file.name}")
                    else:
                        st.warning(f"⚠️ No data extracted from {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Failed to parse {uploaded_file.name}: {e}")
                finally:
                    os.remove(save_path)

        if all_data:
            parsed_df = pd.concat(all_data, ignore_index=True)
            parsed_df = clean_sales_data(parsed_df)

            if st.button("📤 Load Parsed Data to Database"):
                try:
                    with st.spinner("Loading data into database..."):
                        insert_sales_to_db(parsed_df)
                    st.success(f"✅ Loaded {len(parsed_df)} rows into the database.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to load data: {e}")

    st.subheader("📊 Current Housing Dataset")
    if not df_all.empty:
        st.dataframe(df_all, use_container_width=True)
    else:
        st.info("No data available yet. Upload PDFs to get started.")

# =============================
# 📊 TAB 2: CMA TOOL
# =============================
with tab2:
    st.header("📊 Comparative Market Analysis Tool")
    if df_all.empty:
        st.warning("Please upload and load data first.")
    else:
        df_all['Select'] = False
        selected_rows = st.data_editor(
            df_all[[
                'Select', 'mls_number', 'neighborhood', 'house_type', 'garage_type',
                'sqft', 'bedrooms', 'bathrooms', 'built_year', 'age',
                'sold_price', 'list_price', 'sell_list_ratio', 'season'
            ]],
            use_container_width=True,
            key="cma_editor"
        )
        selected_cma = selected_rows[selected_rows['Select']]

        if not selected_cma.empty:
            selected_cma['price_per_sqft'] = (selected_cma['sold_price'] / selected_cma['sqft']).round(2)
            st.dataframe(selected_cma, use_container_width=True)
            st.download_button(
                "📥 Download CMA Report",
                selected_cma.to_csv(index=False),
                "cma_report.csv",
                "text/csv"
            )
        else:
            st.info("Select listings above to generate a CMA report.")

# =============================
# 💰 TAB 3: PRICE PREDICTION
# =============================
with tab3:
    st.header("💰 Predict Winning Offer Price")

    if df_all.empty or model is None:
        st.error("No data or trained model available. Upload and parse data, then train your model first.")
    else:
        default = df_all.iloc[0]
        col1, col2, col3 = st.columns(3)

        with col1:
            neighborhood = st.selectbox(
                "Neighborhood",
                sorted(df_all['neighborhood'].dropna().unique()),
                index=0
            )
            house_type = st.selectbox(
                "House Type",
                sorted(df_all['house_type'].dropna().unique()),
                index=0
            )
            style = st.selectbox(
                "Style",
                sorted(df_all['style'].dropna().unique()),
                index=0
            )
            garage_type = st.selectbox(
                "Garage Type",
                sorted(df_all['garage_type'].dropna().unique()),
                index=0
            )

        with col2:
            bedrooms = st.number_input(
                "Bedrooms", min_value=0, max_value=10,
                value=int(default.get('bedrooms', 3))
            )
            bathrooms = st.number_input(
                "Bathrooms", min_value=0.5, max_value=6.0,
                value=float(default.get('bathrooms', 2.0))
            )
            sqft = st.number_input(
                "Square Footage", min_value=300, max_value=6000,
                value=int(default.get('sqft', 1200))
            )

        with col3:
            built_year = int(default.get('built_year', 2000))
            built_year = built_year if 1900 <= built_year <= datetime.now().year else 2000
            built_year = st.number_input(
                "Built Year", min_value=1900, max_value=datetime.now().year,
                value=built_year
            )
            season = st.selectbox(
                "Season",
                ['Winter', 'Spring', 'Summer', 'Fall'],
                index=2
            )
            list_price = st.number_input(
                "List Price", min_value=50000, max_value=2000000,
                value=int(default.get('list_price', 300000))
            )

       

        if st.button("🔮 Predict Winning Offer Price"):
            try:
                age = datetime.now().year - built_year
                price_diff = 0
                over_asking_pct = 0
                price_per_sqft = list_price / max(sqft, 1)
                # dom_bucket remains, but actual dom is removed

                input_df = pd.DataFrame([{
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'sqft': sqft,
                    'built_year': built_year,
                    'age': age,
                    # 'dom': dom,  # Removed
                    'list_price': list_price,
                    'price_diff': price_diff,
                    'over_asking_pct': over_asking_pct,
                    'price_per_sqft': price_per_sqft,
                    'neighborhood_hotness': 0.5,
                    'realtor_logic': 0.5,
                    'recency_weight': 1,
                    'house_type': house_type,
                    'style': style,
                    'garage_type': garage_type,
                    'season': season,
                    'dom_bucket': '8-14',  # kept if used in pipeline
                    'neighborhood': neighborhood
                }])

                # Ensure all columns expected by the model exist
                for col in model.feature_names_in_:
                    if col not in input_df.columns:
                        input_df[col] = 0

                # Reorder columns to model expected order
                input_df = input_df[model.feature_names_in_]

                # Enforce dtypes
                numeric_cols = [
                    'bedrooms', 'bathrooms', 'sqft', 'built_year', 'age',
                    'list_price', 'price_diff', 'over_asking_pct', 'price_per_sqft',
                    'neighborhood_hotness', 'realtor_logic', 'recency_weight', 'dom'  # if model expects dom
                ]
                for col in numeric_cols:
                    if col in input_df.columns:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

                categorical_cols = [
                    'house_type', 'style', 'garage_type', 'season', 'dom_bucket', 'neighborhood'
                ]
                for col in categorical_cols:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(str)

                input_df = input_df.fillna(0)

                st.write("✅ Cleaned input_df:")
                st.dataframe(input_df)

                predicted_price = model.predict(input_df)[0]
                buffer = max(5000, min(abs(predicted_price * 0.02), 10000))

                st.session_state["predicted_price"] = predicted_price
                st.session_state["buffer"] = buffer

            except Exception as e:
                st.error(f"Prediction failed: {e}")



        if "predicted_price" in st.session_state:
            predicted_price = st.session_state["predicted_price"]
            buffer = st.session_state["buffer"]
            st.success(f"🏆 Recommended Winning Offer: ${predicted_price:,.0f} ±${buffer:,.0f}")

            if st.toggle("Show Feature Importance"):
                try:
                    model_obj = model.named_steps['model'] if 'model' in model.named_steps else model
                    if hasattr(model_obj, 'feature_importances_'):
                        st.subheader("🔍 Top Predictive Features")
                        feature_names = model.named_steps['prep'].get_feature_names_out()
                        feature_imp_series = pd.Series(model_obj.feature_importances_, index=feature_names)
                        top_features = feature_imp_series.abs().sort_values(ascending=False).head(15).reset_index()
                        top_features.columns = ['Feature', 'Importance']
                        fig = px.bar(
                            top_features,
                            x='Importance', y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Blues',
                            height=600
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model.")
                except Exception as e:
                    st.error(f"Error displaying feature importance: {e}")