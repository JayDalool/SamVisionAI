# file: app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import ast
from sqlalchemy import create_engine

# Fix module path to access utils
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="SamVision AI", layout="wide")

st.title("🏡 SamVision AI - Real Estate Analyzer")

pdf_dir = "pdf_uploads"
parsed_csv_path = "parsed_csv/merged.csv"
model_path = "trained_price_model.pkl"
model = joblib.load(model_path) if os.path.exists(model_path) else None

with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

engine = create_engine(
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)



def calculate_features(df):
    df['price_per_sqft'] = (df['sold_price'] / df['sqft']).round(2)
    df['age'] = datetime.now().year - df.get('built_year', datetime.now().year)
    return df

def insert_sales_to_db(df, db_config):
    df.to_sql('housing_data', con=engine, if_exists='append', index=False)

def clean_sales_data(df):
    df = df.drop_duplicates(subset=['mls_number', 'address'], keep='last')
    df = df[df['list_price'] > 0]
    return df

def load_data():
    return pd.read_sql_query("SELECT * FROM housing_data", engine)

df_all = load_data()

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 CMA Tool", "💰 Price Prediction"])

with tab1:
    st.subheader("📥 Upload & Parse PDF Listings")
    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        all_data = []
        for uploaded_file in uploaded_files:
            save_path = os.path.join(pdf_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            st.write(f"Parsing: {uploaded_file.name}")
            try:
                df = extract_pdf_sales(save_path)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                st.error(f"Failed to parse {uploaded_file.name}: {e}")

        if all_data and st.button("📤 Load Parsed Data to DB"):
            parsed_df = pd.concat(all_data, ignore_index=True)
            parsed_df = clean_sales_data(parsed_df)
            insert_sales_to_db(parsed_df, db_config)
            df_all = load_data()
            st.success("Data loaded into database.")

    st.header("📊 Current Housing Dataset")
    if not df_all.empty:
        st.dataframe(df_all)
    else:
        st.info("No data available yet. Upload PDFs to begin.")

with tab2:
    st.header("📊 Comparative Market Analysis")
    if df_all.empty:
        st.warning("No data available. Please upload and load PDF sales.")
    else:
        df_all['Select'] = False
        selected_rows = st.data_editor(
            df_all[['Select', 'neighborhood', 'house_type', 'garage_type', 'sqft',
                    'bedrooms', 'bathrooms', 'sold_price']],
            use_container_width=True, key="cma_editor"
        )
        selected_cma = selected_rows[selected_rows['Select']]
        if not selected_cma.empty:
            selected_cma['price_per_sqft'] = (selected_cma['sold_price'] / selected_cma['sqft']).round(2)
            st.dataframe(selected_cma)
            st.download_button("📥 Download CMA", selected_cma.to_csv(index=False), "cma_report.csv", "text/csv")

with tab3:
    st.header("💰 Predict House Price")
    if df_all.empty or model is None:
        st.error("No data available or model missing. Upload and parse first.")
    else:
        default = df_all.iloc[0]
        default_sqft = int(default.get('sqft', 1200))
        default_sqft = default_sqft if default_sqft >= 600 else 600

        col1, col2, col3 = st.columns(3)
        with col1:
            address = st.text_input("Address", default['address'])
            neighborhood = st.selectbox("Neighborhood", sorted(df_all['neighborhood'].dropna().unique()))
            house_type = st.selectbox("House Type", sorted(df_all['house_type'].dropna().unique()))

        with col2:
            bedrooms = st.slider("Bedrooms", 1, 6, int(default['bedrooms']))
            bathrooms = st.slider("Bathrooms", 1, 4, int(default['bathrooms']))
            sqft = st.number_input("Sqft", min_value=600, max_value=5000, value=default_sqft)

        with col3:
            built_year = st.slider("Built Year", 1900, datetime.now().year, int(default.get('built_year', 2000)))
            garage_type = st.selectbox("Garage Type", sorted(df_all['garage_type'].dropna().unique()))
            season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

        dom = st.slider("🏷️ Days on Market (DOM)", 0, 180, int(default.get('dom', 15)))
        age = datetime.now().year - built_year
        list_price = st.number_input("List Price", min_value=0, value=int(default.get('list_price', 0)))
        price_diff = 0
        over_asking_pct = 0
        price_per_sqft = list_price / max(sqft, 1)
        dom_bucket = pd.cut(pd.Series([dom]), bins=[-1, 7, 14, 30, 90, 180], labels=['0-7', '8-14', '15-30', '31-90', '90+'])[0]

        if st.button("🔮 Predict Price"):
            df_input = pd.DataFrame([{
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft': sqft,
                'age': age,
                'dom': dom,
                'list_price': list_price,
                'price_diff': price_diff,
                'over_asking_pct': over_asking_pct,
                'price_per_sqft': price_per_sqft,
                'neighborhood_hotness': 0.5,
                'house_type': house_type,
                'garage_type': garage_type,
                'season': season,
                'dom_bucket': dom_bucket,
                'neighborhood': neighborhood
            }])

            df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
            predicted_total = model.predict(df_input)[0]
            predicted_overpay = predicted_total - list_price
            predicted_price = list_price + predicted_overpay
            buffer = max(5000, min(abs(predicted_overpay * 0.5), 10000))

            st.markdown(f"<h2 style='color:green;'>🏆 Winning Offer Price: ${predicted_price:,.0f}</h2>", unsafe_allow_html=True)
            st.caption(f"💡 Offer ${predicted_overpay:,.0f} over asking. Confidence ±${buffer:,.0f}")
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
