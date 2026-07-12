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

from utils.auth import (
    auth_config_errors,
    clear_authentication,
    is_authenticated,
    load_auth_config,
    mark_authenticated,
    validate_credentials,
)


def require_login():
    auth_config = load_auth_config()
    errors = auth_config_errors(auth_config)

    if not errors and is_authenticated(st.session_state, auth_config):
        with st.sidebar:
            st.caption(f"Signed in as {auth_config.username}")
            if st.button("Logout", use_container_width=True):
                clear_authentication(st.session_state)
                st.rerun()
        return

    st.title("SamVision AI")
    st.subheader("Sign in")

    if errors:
        st.error(
            "Authentication is not configured. Set SAMVISION_ADMIN_USERNAME, "
            "SAMVISION_ADMIN_PASSWORD_HASH, and SAMVISION_SESSION_SECRET before using the app."
        )
        for error in errors:
            st.warning(error)
        st.stop()

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if validate_credentials(username, password, auth_config):
            mark_authenticated(
                st.session_state,
                auth_config.username,
                auth_config.session_secret,
            )
            st.session_state.pop("samvision_login_failed", None)
            st.rerun()
        st.session_state["samvision_login_failed"] = True

    if st.session_state.get("samvision_login_failed"):
        st.error("Invalid username or password.")

    st.stop()


require_login()

# --- Normal imports ---
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import plotly.express as px  # (you use it elsewhere)

from utils.db_config import get_db_config
from utils.pdf_sales_parser import extract_pdf_sales
from utils.prediction_form import (
    KEY_BASEMENT_TYPE,
    KEY_BATHROOMS,
    KEY_BEDROOMS,
    KEY_BUILT_YEAR,
    KEY_GARAGE_TYPE,
    KEY_HOUSE_TYPE,
    KEY_IS_MULTI_OFFER,
    KEY_LIST_PRICE,
    KEY_NEIGHBORHOOD,
    KEY_PREMIUM_PCT,
    KEY_SEASON,
    KEY_SQFT,
    KEY_STYLE,
    BASEMENT_TYPES,
    SEASONS,
    basement_adjustment,
    field_defaults,
    init_prediction_state,
    reset_prediction_state,
    suggested_multi_offer_premium,
)

# ✅ NEW: Rental Income tool tab
from utils.rental_income_tool import render_rental_income_tab

# --- CONFIG ---
pdf_dir = "pdf_uploads"
model_path = "trained_price_model.pkl"
os.makedirs(pdf_dir, exist_ok=True)

# Load trained model
model = joblib.load(model_path) if os.path.exists(model_path) else None


# --- DB ENGINE ---
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = get_db_config()
    url = URL.create(
        "postgresql+psycopg2",
        username=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["dbname"],
    )
    return create_engine(url)


# --- UTILS ---
def insert_sales_to_db(df: pd.DataFrame):
    # ✅ Fix: ensure engine exists
    engine = get_engine()
    df.to_sql("housing_data", con=engine, if_exists="append", index=False)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "mls_number" in df.columns and "address" in df.columns:
        df = df.drop_duplicates(subset=["mls_number", "address"], keep="last")
    if "list_price" in df.columns:
        df = df[df["list_price"] > 0]
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
st.markdown(
    """
**Workflow for Realtors:**
1️⃣ **Upload MLS PDFs to auto-extract sold listings**  
2️⃣ **Load parsed data into your database** for analysis  
3️⃣ Generate **Comparative Market Analyses (CMA) easily**  
4️⃣ Predict **winning offer prices** within ±$10,000 for competitive listings
"""
)

df_all = load_data()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["📥 Upload & Parse", "📊 CMA Tool", "💰 Price Prediction", "🏠 Rental Income"]
)

# =============================
# 📥 TAB 1: Upload & Parse PDFs
# =============================
with tab1:
    st.header("📥 Upload & Parse MLS PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDF Files", type=["pdf"], accept_multiple_files=True
    )

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
                    # remove uploaded temp file
                    if os.path.exists(save_path):
                        os.remove(save_path)

        if all_data:
            parsed_df = pd.concat(all_data, ignore_index=True)
            parsed_df = clean_sales_data(parsed_df)

            if st.button("📤 Load Parsed Data to Database"):
                try:
                    with st.spinner("Loading data into database..."):
                        insert_sales_to_db(parsed_df)
                    st.success(f"✅ Loaded {len(parsed_df)} rows into the database.")
                    st.cache_data.clear()
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
        df_view = df_all.copy()
        df_view["Select"] = False

        cols = [
            "Select",
            "mls_number",
            "neighborhood",
            "house_type",
            "garage_type",
            "sqft",
            "bedrooms",
            "bathrooms",
            "built_year",
            "age",
            "sold_price",
            "list_price",
            "sell_list_ratio",
            "season",
        ]
        cols = [c for c in cols if c in df_view.columns]

        selected_rows = st.data_editor(
            df_view[cols],
            use_container_width=True,
            key="cma_editor",
        )

        selected_cma = selected_rows[selected_rows["Select"] == True]

        if not selected_cma.empty:
            if "sqft" in selected_cma.columns and "sold_price" in selected_cma.columns:
                selected_cma = selected_cma.copy()
                selected_cma["price_per_sqft"] = (
                    selected_cma["sold_price"] / selected_cma["sqft"]
                ).round(2)

            st.dataframe(selected_cma, use_container_width=True)
            st.download_button(
                "📥 Download CMA Report",
                selected_cma.to_csv(index=False),
                "cma_report.csv",
                "text/csv",
            )
        else:
            st.info("Select listings above to generate a CMA report.")

# =============================
# 💰 TAB 3: PRICE PREDICTION
# =============================
with tab3:
    st.header("💰 Predict Winning Offer Price")

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

    st.warning(
        "⚠️ Disclaimer: This tool provides an educated estimate and does not guarantee pricing. "
        "Always verify with current market data and professional judgment."
    )

    if df_all.empty:
        st.warning("No data loaded yet. Upload & load data in tab 1 first.")
        st.stop()

    default = df_all.iloc[0]

    # Defensive unique lists
    def uniq(col, fallback):
        if col in df_all.columns:
            vals = sorted(df_all[col].dropna().unique())
            return vals if vals else fallback
        return fallback

    neighborhood_options = uniq("neighborhood", ["Unknown"])
    house_type_options = uniq("house_type", ["Unknown"])
    style_options = uniq("style", ["Unknown"])
    garage_type_options = uniq("garage_type", ["Unknown"])

    # Seed widget state once (never overwrites existing user input). Every widget
    # below uses a stable explicit key, so a rerun triggered by editing one field
    # never resets the others.
    init_prediction_state(
        st.session_state,
        neighborhoods=neighborhood_options,
        house_types=house_type_options,
        styles=style_options,
        garage_types=garage_type_options,
        defaults=field_defaults(default),
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        neighborhood = st.selectbox("Neighborhood", neighborhood_options, key=KEY_NEIGHBORHOOD)
        house_type = st.selectbox("House Type", house_type_options, key=KEY_HOUSE_TYPE)
        style = st.selectbox("Style", style_options, key=KEY_STYLE)
        garage_type = st.selectbox("Garage Type", garage_type_options, key=KEY_GARAGE_TYPE)

    with col2:
        basement_type = st.selectbox("Basement Type", BASEMENT_TYPES, key=KEY_BASEMENT_TYPE)
        bedrooms = st.number_input("Bedrooms", 0, 10, step=1, key=KEY_BEDROOMS)
        bathrooms = st.number_input("Bathrooms", 0.5, 6.0, step=0.5, key=KEY_BATHROOMS)
        sqft = st.number_input("Square Footage", 300, 6000, step=50, key=KEY_SQFT)

    with col3:
        built_year = st.number_input("Built Year", 1900, datetime.now().year, step=1, key=KEY_BUILT_YEAR)
        season = st.selectbox("Season", SEASONS, key=KEY_SEASON)
        list_price = st.number_input("List Price", 50000, 2000000, step=1000, key=KEY_LIST_PRICE)
        is_multi_offer = st.checkbox("🏠 Multi-Offer Listing", key=KEY_IS_MULTI_OFFER)

    st.subheader("🔥 Adjust Multi-Offer Premium (%)")
    # The slider owns its own value via a stable key; it is NOT re-defaulted on
    # every rerun (that was the bug that reset it when other fields changed). The
    # neighborhood suggestion is applied only when the user clicks the button.
    suggested_premium = suggested_multi_offer_premium(neighborhood, is_multi_offer)

    def _apply_suggested_premium():
        st.session_state[KEY_PREMIUM_PCT] = suggested_multi_offer_premium(
            st.session_state.get(KEY_NEIGHBORHOOD),
            st.session_state.get(KEY_IS_MULTI_OFFER, False),
        )

    prem_col, btn_col = st.columns([3, 1])
    with prem_col:
        user_premium_pct = st.slider("Expected Multi-Offer Premium (%)", 0.0, 25.0, step=0.5, key=KEY_PREMIUM_PCT)
    with btn_col:
        if is_multi_offer:
            st.caption(f"Suggested for {neighborhood}: {suggested_premium:.1f}%")
            st.button("Use suggested", on_click=_apply_suggested_premium, use_container_width=True)

    st.button(
        "♻️ Reset Form",
        on_click=reset_prediction_state,
        args=(st.session_state,),
        help="Clear all prediction inputs back to their defaults.",
    )

    if st.button("🔮 Predict Winning Offer Price"):
        if model is None:
            st.error("No trained model available. Please train or load a model first.")
        else:
            try:
                age = datetime.now().year - built_year
                price_per_sqft = list_price / max(sqft, 1)
                premium_multiplier = 1 + user_premium_pct / 100 if is_multi_offer else 1
                basement_adj = basement_adjustment(basement_type)

                input_df = pd.DataFrame([{
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "sqft": sqft,
                    "built_year": built_year,
                    "age": age,
                    "list_price": list_price,
                    "price_diff": 0,
                    "over_asking_pct": 0,
                    "price_per_sqft": price_per_sqft,
                    "neighborhood_hotness": 0.6 if is_multi_offer else 0.5,
                    "realtor_logic": 0.8 if is_multi_offer else 0.5,
                    "recency_weight": 1,
                    "multi_offer_flag": int(is_multi_offer),
                    "likely_multi_offer": int(is_multi_offer),
                    "season_boost": 1.05 if season == "Summer" else 1.0,
                    "comp_count_in_neighborhood": df_all[df_all["neighborhood"] == neighborhood].shape[0] if "neighborhood" in df_all.columns else 0,
                    "house_type": house_type,
                    "garage_type": garage_type,
                    "season": season,
                    "neighborhood": neighborhood,
                    "style": style,
                }])

                # Align with model features if available
                if hasattr(model, "feature_names_in_"):
                    for col in model.feature_names_in_:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    input_df = input_df[model.feature_names_in_].fillna(0)

                predicted_price = model.predict(input_df)[0] * premium_multiplier + basement_adj
                buffer = max(5000, min(abs(predicted_price * 0.02), 10000))

                st.success(f"🏆 Recommended Winning Offer: ${predicted_price:,.0f} ±${buffer:,.0f}")
                st.info(
                    f"Strategy: {'Multi-Offer' if is_multi_offer else 'Standard'} | "
                    f"Premium: {user_premium_pct:.1f}% | Basement Adj: {basement_adj:+,}"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# =============================
# 🏠 TAB 4: RENTAL INCOME TOOL
# =============================
with tab4:
    render_rental_income_tab()
