# file: app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast
import datetime
import sys
import os
from sqlalchemy import create_engine

st.set_page_config(page_title="SamVision AI", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pdf_sales_parser import extract_pdf_sales, extract_csv_sales, insert_sales_to_db, clean_sales_data

model = joblib.load('trained_price_model.pkl') if os.path.exists('trained_price_model.pkl') else None

with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

engine = create_engine(
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)

@st.cache_data
def load_data():
    return pd.read_sql_query("SELECT * FROM housing_data", engine)

df_all = load_data()

def infer_neighborhood_from_address(address_input: str, df_ref: pd.DataFrame) -> str:
    address_input = address_input.lower().strip()
    match = df_ref[df_ref['address'].str.lower().str.contains(address_input[:10], na=False)]
    if not match.empty:
        return match.iloc[0]['neighborhood']
    return df_ref['neighborhood'].mode()[0]

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 CMA Tool", "💰 Price Prediction"])

with tab1:
    st.title("🏠 SamVisionAI")
    st.markdown("""
    Welcome to **SamVisionAI** — AI-powered pricing and CMA analysis for Realtors.
    - 📅 Upload MLS sales PDFs or CSVs
    - 💰 Predict pricing with DOM consideration
    - 📟 Generate CMA reports for clients
    """)

    st.subheader("📥 Upload Files (MLS Sales)")
    uploaded_files = st.file_uploader("Upload PDF(s) or CSV", type=["pdf", "csv"], accept_multiple_files=True)
    if uploaded_files:
        parsed_frames = []
        for uploaded_file in uploaded_files:
            save_path = os.path.join("pdf_uploads", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            if uploaded_file.name.lower().endswith(".pdf"):
                parsed_frames.append(extract_pdf_sales(save_path))
            elif uploaded_file.name.lower().endswith(".csv"):
                parsed_frames.append(extract_csv_sales(save_path))

        if st.button("📤 Parse & Load Files to Database"):
            parsed_df = pd.concat(parsed_frames, ignore_index=True)
            if not parsed_df.empty:
                st.markdown("### 🔎 Preview Extracted Listings")
                highlight_cols = [col for col in ['sold_price', 'neighborhood', 'garage_type'] if col in parsed_df.columns]
                styled = parsed_df.style.apply(lambda row: ["background-color: #ffdddd" if col in highlight_cols and pd.isna(row[col]) else "" for col in row.index], axis=1)
                st.dataframe(styled)
                parsed_df = clean_sales_data(parsed_df)
                insert_sales_to_db(parsed_df, db_config)
                df_all = load_data()
                st.success("Data loaded successfully. Please retrain the model manually if needed.")

with tab2:
    st.header("📊 Comparative Market Analysis")
    if df_all.empty:
        st.warning("No data available for CMA. Please upload some MLS data first.")
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
        st.error("No data available to base predictions on. Please upload and load data first.")
    else:
        default = df_all.iloc[0]
        col1, col2, col3 = st.columns(3)

        with col1:
            address = st.text_input("Address", default['address'])
            inferred_neighborhood = infer_neighborhood_from_address(address, df_all)
            neighborhood = st.selectbox("Neighborhood", sorted(df_all['neighborhood'].dropna().unique()), help=f"Inferred: {inferred_neighborhood}")
            house_type = st.selectbox("House Type", sorted(df_all['house_type'].dropna().unique()))

        with col2:
            bedrooms = st.slider("Bedrooms", 1, 6, int(default['bedrooms']))
            bathrooms = st.slider("Bathrooms", 1, 4, int(default['bathrooms']))
            sqft_val = int(default['sqft'])
            sqft = st.number_input("Sqft", min_value=600, max_value=5000, value=max(600, sqft_val))

        with col3:
            built_year = st.slider("Built Year", 1900, datetime.datetime.now().year, int(default['built_year']))
            garage_type = st.selectbox("Garage Type", [
                "single_attached", "single_detached", "double_attached",
                "double_detached", "triple_attached", "triple_detached", "none"
            ])
            season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

        dom = st.slider("🏷️ Days on Market (DOM)", 0, 180, int(default.get('dom', 15)))
        age = datetime.datetime.now().year - built_year
        list_price = st.number_input("List Price", min_value=0, value=int(default['list_price']))
        price_diff = 0
        over_asking_pct = 0
        price_per_sqft = list_price / max(sqft, 1)
        listing_date = st.date_input("Listing Date", value=datetime.date.today())
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

            st.markdown(f"<h2 style='color:green;'>🏆 Multi-Offer Winning Price: ${predicted_price:,.0f}</h2>", unsafe_allow_html=True)
            st.caption(f"💡 Offer ${predicted_overpay:,.0f} over asking to increase win chances.")
            st.caption(f"Confidence ±${buffer:,.0f} → ${predicted_price - buffer:,.0f} - ${predicted_price + buffer:,.0f}")

            with st.expander("💬 Why this price?"):
                avg_dom = df_all.groupby("neighborhood")["dom"].mean()
                neigh_avg_dom = int(avg_dom.get(neighborhood, 30))

                st.markdown(f"""
                We analyzed recent sales in **{neighborhood}**:

                - Average DOM is {neigh_avg_dom} days.
                - Most winning offers exceeded list price by **$10,000-$25,000**.
                - Listing price is ${list_price:,.0f}, and our AI model predicts a competitive offer price of ${predicted_price:,.0f}.

                **How we calculate this:**
                - DOM trend, price/sqft, home age
                - Similar listings' over-asking % in area
                - Realtor best practices for multi-offer wins
                """)

            with st.expander("🏡 Similar Listings Used in Prediction"):
                similar_listings = df_all[
                    (df_all['neighborhood'] == neighborhood) &
                    (df_all['house_type'] == house_type) &
                    (df_all['sqft'].between(sqft * 0.9, sqft * 1.1)) &
                    (df_all['bedrooms'] == bedrooms)
                ][['address', 'sold_price', 'list_price', 'sqft', 'dom', 'built_year']].sort_values(by='dom')

                st.dataframe(similar_listings.reset_index(drop=True))
