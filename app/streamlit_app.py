# file: app/streamlit_app.py

import streamlit as st

st.set_page_config(
    page_title="🏡 SamVision AI - Realtor CMA + Multi-Offer Predictor",
    layout="wide",
)

# --- Path setup: make project root importable ---
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --- Normal imports ---
import pandas as pd
import numpy as np
import ast
import joblib
from datetime import datetime
import json

from sqlalchemy import create_engine
import plotly.express as px

# ✅ your actual config lives here:
from utils.db_config import get_db_config

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
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = get_db_config()   # SAME config as load_to_postgresql.py
    url = (
        f"postgresql+psycopg2://"
        f"{cfg['user']}:{cfg['password']}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )
    return create_engine(url)



# --- UTILS ---
def insert_sales_to_db(df):
    df.to_sql('housing_data', con=engine, if_exists='append', index=False)

def clean_sales_data(df):
    df = df.drop_duplicates(subset=['mls_number', 'address'], keep='last')
    df = df[df['list_price'] > 0]
    return df

@st.cache_data(show_spinner=False)
def load_data():
    engine = get_engine()
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

with tab3:
    st.header("💰 Predict Winning Offer Price")

    # Educated guess premiums (all other areas will default to 5%)
    multi_offer_premiums = {
        # Hot competition zones
        "Bridgwater Trails": 8.0, "Bridgwater Lakes": 7.5, "Bridgwater Forest": 7.5, "Bridgwater Centre": 7.5,
        "Tuxedo": 9.0, "Old Tuxedo": 9.0, "Crescentwood": 8.0, "Crescent Park": 8.0,
        "Osborne Village": 8.5, "River-Osborne": 8.0, "West Wolseley": 7.5,
        "Waverley Heights": 7.0, "Waverley West B": 7.0, "Fort Richmond": 7.0, "Linden Woods": 8.0,

        # Mid-tier competition
        "River Heights": 6.5, "North River Heights": 6.5, "Sage Creek": 6.5,
        "Richmond West": 6.5, "Grant Park": 6.0, "Jameswood": 6.0,
        "Kensington": 6.0, "King Edward": 6.0, "University": 6.0,
        "Central River Heights": 6.5, "Exchange District": 6.0, "Prairie Pointe": 6.0,

        # Balanced suburbs
        "Charleswood": 5.5, "St. Vital": 5.5, "St. Norbert": 5.5,
        "Amber Trails": 5.5, "Garden City": 5.5, "Maples": 5.5,
        "Westwood": 5.5, "Elm Park": 5.5
        # All others default to 5.0%
    }

    # Load model explanation
    explanation = {}
    try:
        with open("model_explanation.json", "r") as f:
            explanation = json.load(f)
    except Exception:
        explanation = {"note": "Model explanation not found. Retrain to generate explanations."}

    with st.container(border=True):
        st.subheader("🩺 Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"✅ **Model:** {explanation.get('model_used', 'Unknown')}")
            st.markdown(f"📅 **Trained on:** {explanation.get('trained_on_rows', 'N/A')} rows")
            st.markdown(f"🎯 **MAE:** ${explanation.get('mae', 0):,.0f}")
            st.markdown(f"📈 **R²:** {explanation.get('r2', 0):.4f}")
        with col2:
            st.markdown("🛠️ **Features Used:**")
            features_list = explanation.get("features_used", [])
            if features_list:
                st.markdown(", ".join(features_list[:10]) + ("..." if len(features_list) > 10 else ""))
            else:
                st.markdown("_Features unavailable._")
        st.info(explanation.get("note", "No detailed notes provided."))

    st.info(
        "💡 **How to use:** Enter property details below. Check Multi-Offer to auto-fill local premium. "
        "Adjust manually if needed for accurate winning offer price predictions."
    )

    # Disclaimer
    st.warning("⚠️ Disclaimer: This tool provides an *educated estimate* based on historical stats and does not guarantee precise pricing. Always verify with current market data and professional judgment.")
    
    if df_all.empty:
        st.warning("No data loaded yet. Upload & load data in tab 1 first.")
        st.stop()

    # 🏠 Inputs
    default = df_all.iloc[0]
    col1, col2, col3 = st.columns(3)

    with col1:
        neighborhood = st.selectbox("Neighborhood", sorted(df_all['neighborhood'].dropna().unique()), index=0)
        house_type = st.selectbox("House Type", sorted(df_all['house_type'].dropna().unique()), index=0)
        style = st.selectbox("Style", sorted(df_all['style'].dropna().unique()), index=0)
        garage_type = st.selectbox("Garage Type", sorted(df_all['garage_type'].dropna().unique()), index=0)

    with col2:
        basement_type = st.selectbox("Basement Type",["None", "Crawl Space", "Full (Unfinished)", "Full (Finished)", "Walkout"],index=0)
        bedrooms = st.number_input("Bedrooms", 0, 10, int(default.get('bedrooms', 3)))
        bathrooms = st.number_input("Bathrooms", 0.5, 6.0, float(default.get('bathrooms', 2.0)))
        sqft = st.number_input("Square Footage", 300, 6000, int(default.get('sqft', 1200)))

    with col3:
        built_year = st.number_input("Built Year", 1900, datetime.now().year, int(default.get('built_year', 2000)))
        season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'], index=2)
        list_price = st.number_input("List Price", 50000, 2000000, int(default.get('list_price', 300000)))
        is_multi_offer = st.checkbox("🏠 Multi-Offer Listing", value=False)

    

    # 🔥 Multi-Offer Premium Slider
    st.subheader("🔥 Adjust Multi-Offer Premium (%)")
    st.caption(
        "This is the percentage buyers typically pay above list price in multi-offer situations.\n"
        "- **5–9%**: Normal Winnipeg markets\n"
        "- **10–15%**: Rare aggressive underpricing\n"
        "- **20%+**: Extremely rare; adjust manually\n"
        "Adjust based on your local market knowledge."
    )

    default_premium = multi_offer_premiums.get(neighborhood or "", 5.0) if is_multi_offer else 0
    user_premium_pct = st.slider(
        "Expected Multi-Offer Premium (%)",
        0.0, 25.0, float(default_premium), 0.5
    )

    # 🚀 Prediction Logic
    if st.button("🔮 Predict Winning Offer Price"):
        if model is None:
            st.error("No trained model available. Please train or load a model first.")
        else:
            try:
                age = datetime.now().year - built_year
                price_per_sqft = list_price / max(sqft, 1)
                premium_multiplier = 1 + user_premium_pct / 100 if is_multi_offer else 1
                basement_adj = {"None": -45000, "Crawl Space": -45000, "Full (Unfinished)": -15000,
                                "Full (Finished)": 25000, "Walkout": 0}.get(basement_type, 0)

                if basement_type == "Walkout":
                    st.warning("⚠️ Walkout basements may significantly impact valuation. Manual review recommended.")

                input_df = pd.DataFrame([{
                    'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft': sqft,
                    'built_year': built_year, 'age': age, 'list_price': list_price,
                    'price_diff': 0, 'over_asking_pct': 0, 'price_per_sqft': price_per_sqft,
                    'neighborhood_hotness': 0.6 if is_multi_offer else 0.5,
                    'realtor_logic': 0.8 if is_multi_offer else 0.5, 'recency_weight': 1,
                    'multi_offer_flag': int(is_multi_offer), 'likely_multi_offer': int(is_multi_offer),
                    'season_boost': 1.05 if season == 'Summer' else 1.0,
                    'comp_count_in_neighborhood': df_all[df_all['neighborhood'] == neighborhood].shape[0],
                    'house_type': house_type, 'garage_type': garage_type,
                    'season': season, 'neighborhood': neighborhood, 'style': style
                }])

                for col in getattr(model, "feature_names_in_", []):
                    if col not in input_df.columns:
                        input_df[col] = 0

                if hasattr(model, "feature_names_in_"):
                    input_df = input_df[model.feature_names_in_].fillna(0)
                    predicted_price = model.predict(input_df)[0] * premium_multiplier + basement_adj
                    buffer = max(5000, min(abs(predicted_price * 0.02), 10000))

                    st.success(f"🏆 Recommended Winning Offer: ${predicted_price:,.0f} ±${buffer:,.0f}")
                    st.info(
                        f"✅ Using: {'Multi-Offer' if is_multi_offer else 'Standard'} Strategy | "
                        f"Premium: {user_premium_pct:.1f}% | Basement Adj: {basement_adj:+,}"
                    )

                    st.session_state["predicted_price"] = predicted_price
                    st.session_state["buffer"] = buffer
                else:
                    st.error("Loaded model does not have feature_names_in_. Please retrain or check your pipeline.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
