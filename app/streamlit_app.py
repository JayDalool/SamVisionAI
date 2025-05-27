# file: app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import psycopg2
import ast
import datetime
import tempfile
import sys
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pdf_sales_parser import extract_pdf_sales, insert_sales_to_db
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = joblib.load('trained_price_model.pkl')
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

conn = psycopg2.connect(**db_config)
df_all = pd.read_sql_query("SELECT * FROM housing_data", conn)
conn.close()

lot_size_sqft = float(df_all['lot_size'].median()) if 'lot_size' in df_all.columns else 5000.0

st.set_page_config(page_title="SamVision AI", layout="wide")
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 CMA Tool", "💰 Price Prediction"])

with tab1:
    st.title("🏠 SamVisionAI")
    st.markdown("""
    Welcome to **SamVisionAI** — AI-powered pricing and CMA analysis for Realtors.
    - 📥 Upload MLS sales PDFs
    - 💰 Predict pricing with DOM consideration
    - 🧾 Generate CMA reports for clients
    """)

    st.subheader("📥 Upload PDF Files (MLS Sales)")
    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded_pdfs:
        for uploaded_file in uploaded_pdfs:
            save_path = os.path.join("pdf_uploads", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())

        if st.button("📤 Parse & Load PDFs to Database"):
            parsed_frames = [extract_pdf_sales(os.path.join("pdf_uploads", f.name)) for f in uploaded_pdfs]
            parsed_df = pd.concat(parsed_frames, ignore_index=True)

            if not parsed_df.empty:
                st.markdown("### 🔎 Preview Extracted Listings")

                highlight_cols = [col for col in ['sold_price', 'region', 'neighborhood', 'garage_type'] if col in parsed_df.columns]

                def highlight_nulls_limited(row):
                    return ["background-color: #ffdddd" if col in highlight_cols and pd.isna(row[col]) else "" for col in row.index]

                styled = parsed_df.style.apply(highlight_nulls_limited, axis=1)
                st.dataframe(styled)

                inserted, skipped = insert_sales_to_db(parsed_df, db_config)
                st.success(f"✅ Inserted {inserted} records. Skipped {len(skipped)}.")

                conn = psycopg2.connect(**db_config)
                df = pd.read_sql_query("SELECT * FROM housing_data", conn)
                conn.close()

                df['age'] = datetime.datetime.now().year - df['built_year']
                df = pd.get_dummies(df, columns=['neighborhood', 'region', 'house_type', 'garage_type', 'season'], drop_first=True)
                X = df.drop(['sold_price', 'built_year', 'listing_date', 'latitude', 'longitude', 'address'], axis=1, errors='ignore')
                y = df['sold_price']

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                joblib.dump(model, 'trained_price_model.pkl')
                st.success("✅ Model retrained and updated!")

with tab2:
    st.header("📊 Comparative Market Analysis")
    df_all['Select'] = False
    selected_rows = st.data_editor(
        df_all[['Select', 'neighborhood', 'region', 'house_type', 'garage_type', 'sqft',
                'bedrooms', 'bathrooms', 'lot_size', 'sold_price']],
        use_container_width=True, key="cma_editor"
    )
    selected_cma = selected_rows[selected_rows['Select']]
    if not selected_cma.empty:
        selected_cma['price_per_sqft'] = (selected_cma['sold_price'] / selected_cma['sqft']).round(2)
        st.dataframe(selected_cma)
        st.download_button("📥 Download CMA", selected_cma.to_csv(index=False), "cma_report.csv", "text/csv")

with tab3:
    st.header("💰 Predict House Price")
    default = df_all.iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        neighborhood = st.text_input("Neighborhood", default['neighborhood'])
        region = st.text_input("Region", default['region'])
        house_type = st.selectbox("House Type", sorted(df_all['house_type'].dropna().unique()))

    with col2:
        bedrooms = st.slider("Bedrooms", 1, 6, default['bedrooms'])
        bathrooms = st.slider("Bathrooms", 1, 4, default['bathrooms'])
        sqft = st.number_input("Sqft", 600, 5000, default['sqft'])

    with col3:
        lot_size = st.number_input("Lot Size (sqft)", min_value=0.0, max_value=100000.0, value=round(lot_size_sqft, 2))
        built_year = st.slider("Built Year", 1900, datetime.datetime.now().year, default['built_year'])
        garage_type = st.selectbox("Garage Type", sorted(df_all['garage_type'].dropna().unique()))
        season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

    dom = st.slider("🏷️ Days on Market (DOM)", 0, 180, 15)
    age = datetime.datetime.now().year - built_year

    if st.button("🔮 Predict Price"):
        input_dict = {
            'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft': sqft,
            'lot_size': lot_size, 'age': age, 'dom': dom
        }

        cat_cols = ['neighborhood', 'region', 'house_type', 'season', 'garage_type']
        df_encoded = pd.get_dummies(df_all[cat_cols], drop_first=True)
        for col in df_encoded.columns:
            input_dict[col] = int(
                col.endswith(str(neighborhood or "")) or
                col.endswith(str(region or "")) or
                col.endswith(str(house_type or "")) or
                col.endswith(str(season or "")) or
                col.endswith(str(garage_type or ""))
            )

        df_input = pd.DataFrame([input_dict])
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

        predicted_price = model.predict(df_input)[0]

        conn = psycopg2.connect(**db_config)
        df_eval = pd.read_sql_query("SELECT * FROM housing_data", conn)
        conn.close()

        df_eval['age'] = datetime.datetime.now().year - df_eval['built_year']
        df_eval['dom'] = df_eval.get("dom", pd.Series([15] * len(df_eval)))
        df_eval = pd.get_dummies(df_eval, columns=cat_cols, drop_first=True)
        X_eval = df_eval.drop(['sold_price', 'built_year', 'listing_date', 'latitude', 'longitude'],
                              axis=1, errors='ignore')
        X_eval = X_eval.reindex(columns=model.feature_names_in_, fill_value=0)
        y_eval = df_eval['sold_price']
        preds_eval = model.predict(X_eval)
        mae = mean_absolute_error(y_eval, preds_eval)
        buffer = max(5000, min(mae, 7500))

        st.markdown(f"<h2 style='color:green;'>💰 Predicted Price: ${predicted_price:,.0f}</h2>", unsafe_allow_html=True)
        st.caption(f"Confidence ±${buffer:,.0f} → ${predicted_price - buffer:,.0f} - ${predicted_price + buffer:,.0f}")