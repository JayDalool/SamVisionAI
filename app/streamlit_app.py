# file: app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import psycopg2
import ast
import datetime
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load trained model
model = joblib.load('trained_price_model.pkl')

# Load DB config
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

# Setup layout
st.set_page_config(page_title="🏠 SamVisionAI - Price Predictor", layout="wide")
st.title("🏠 Winnipeg House Price Prediction")

# Load all data
conn = psycopg2.connect(**db_config)
df_all = pd.read_sql_query("SELECT * FROM housing_data", conn)
conn.close()

# Sidebar Filters
with st.sidebar:
    st.header("🔍 Filter Listings")
    search_neighborhood = st.text_input("Search by Neighborhood")
    filter_region = st.selectbox("Filter by Region", ["All"] + sorted(df_all['region'].dropna().unique()))
    filter_type = st.selectbox("House Type", ["All"] + sorted(df_all['house_type'].dropna().unique()))
    filter_garage = st.selectbox("Garage Type", ["All"] + sorted(df_all['garage_type'].dropna().unique()))
    min_bed = st.slider("Min Bedrooms", 1, 6, 1)
    min_bath = st.slider("Min Bathrooms", 1, 4, 1)

# Region selector buttons
st.subheader("🌍 Choose a Region")
cols = st.columns(len(df_all['region'].unique()))
selected_region = st.session_state.get("selected_region")
for i, region in enumerate(sorted(df_all['region'].unique())):
    if cols[i].button(region):
        st.session_state["selected_region"] = region
        selected_region = region

# Determine selected row
if search_neighborhood:
    filtered_df = df_all[df_all['neighborhood'].str.contains(search_neighborhood, case=False)]
    selected_row = filtered_df.iloc[0] if not filtered_df.empty else df_all.iloc[0]
elif selected_region:
    region_df = df_all[df_all['region'] == selected_region]
    selected_row = region_df.iloc[0] if not region_df.empty else df_all.iloc[0]
else:
    selected_row = df_all.iloc[0]

# UI Form
st.subheader("🏘️ House Details")
col1, col2, col3 = st.columns(3)

with col1:
    neighborhood = st.text_input("Neighborhood", selected_row['neighborhood'])
    region = st.text_input("Region", selected_row['region'])
    house_type = st.selectbox("House Type", sorted(df_all['house_type'].unique()), index=0)

with col2:
    bedrooms = st.slider("Bedrooms", 1, 6, selected_row['bedrooms'])
    bathrooms = st.slider("Bathrooms", 1, 4, selected_row['bathrooms'])
    sqft = st.number_input("Square Footage", 600, 5000, selected_row['sqft'])

with col3:
    lot_size = st.number_input("Lot Size (sqft)", 0.0, 10000.0, float(selected_row['lot_size']))
    built_year = st.slider("Built Year", 1900, datetime.datetime.now().year, selected_row['built_year'])
    garage_type = st.selectbox("Garage Type", sorted(df_all['garage_type'].unique()))
    season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

age = datetime.datetime.now().year - built_year
latitude = selected_row['latitude']
longitude = selected_row['longitude']

# Predict
if st.button("Predict Price"):
    input_dict = {
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft': sqft,
        'lot_size': lot_size, 'age': age
    }

    # One-hot encode all categorical
    cat_cols = ['neighborhood', 'region', 'house_type', 'season', 'garage_type']
    df_encoded = pd.get_dummies(df_all[cat_cols], drop_first=True)
    for col in df_encoded.columns:
        input_dict[col] = 1 if (
            col.endswith(neighborhood) or col.endswith(region) or
            col.endswith(house_type) or col.endswith(season) or col.endswith(garage_type)
        ) else 0

    df_input = pd.DataFrame([input_dict])
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
    predicted_price = model.predict(df_input)[0]

    # Error estimate
    conn = psycopg2.connect(**db_config)
    df_eval = pd.read_sql_query("SELECT * FROM housing_data", conn)
    conn.close()
    df_eval['age'] = datetime.datetime.now().year - df_eval['built_year']
    df_eval = pd.get_dummies(df_eval, columns=cat_cols, drop_first=True)
    X_eval = df_eval.drop(['sold_price', 'built_year', 'listing_date', 'latitude', 'longitude'], axis=1, errors='ignore')
    X_eval = X_eval.reindex(columns=model.feature_names_in_, fill_value=0)
    y_eval = df_eval['sold_price']
    preds_eval = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, preds_eval)

    st.markdown(f"<h2 style='color:green;'>💰 Estimated Price: ${predicted_price:,.0f}</h2>", unsafe_allow_html=True)
    st.caption(f"Confidence Range: ${predicted_price - mae:,.0f} - ${predicted_price + mae:,.0f} (±${mae:,.0f})")

    # Save prediction
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO housing_data (
            neighborhood, region, house_type, bedrooms, bathrooms, sqft,
            lot_size, built_year, garage_type, listing_date, season,
            latitude, longitude, sold_price
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        neighborhood, region, house_type, bedrooms, bathrooms, sqft,
        lot_size, built_year, garage_type, datetime.date.today(), season,
        latitude, longitude, int(predicted_price)
    ))
    conn.commit()
    cursor.close()
    conn.close()

    st.success("✅ Prediction saved to database")

# Map & Results
st.subheader("📍 Homes on the Market")
df_filtered = df_all.copy()
if filter_region != "All":
    df_filtered = df_filtered[df_filtered['region'] == filter_region]
if filter_type != "All":
    df_filtered = df_filtered[df_filtered['house_type'] == filter_type]
if filter_garage != "All":
    df_filtered = df_filtered[df_filtered['garage_type'] == filter_garage]
df_filtered = df_filtered[df_filtered['bedrooms'] >= min_bed]
df_filtered = df_filtered[df_filtered['bathrooms'] >= min_bath]

st.dataframe(df_filtered.sort_values("sold_price", ascending=False)[[
    'neighborhood', 'region', 'house_type', 'garage_type', 'sqft',
    'bedrooms', 'bathrooms', 'sold_price'
]].head(25))

# Map
if 'latitude' in df_filtered.columns:
    df_filtered['latlon'] = df_filtered[['longitude', 'latitude']].values.tolist()
    layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_filtered[['longitude', 'latitude']].to_dict(orient='records'),
    get_position='[longitude, latitude]',
    get_color='[100, 200, 255, 160]',
    get_radius=150,
    pickable=True
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=11),
        layers=[layer]
    ))
