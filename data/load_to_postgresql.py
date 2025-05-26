# file: data/load_to_postgresql.py

import os
import sys
import glob
import ast
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pdf_sales_parser import extract_pdf_sales

CSV_PATH = 'winnipeg_housing_data.csv'
PDF_DIR = 'pdf_uploads'

with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

def get_dataframe():
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        try:
            df = pd.read_csv(CSV_PATH)

            # Patch missing lat/lng after reading
            if 'latitude' not in df.columns:
                df['latitude'] = 0.0
            if 'longitude' not in df.columns:
                df['longitude'] = 0.0

            return df
        except pd.errors.EmptyDataError:
            print("⚠️ CSV exists but is empty or malformed. Re-parsing from PDFs...")
            os.remove(CSV_PATH)

    if os.path.exists(PDF_DIR):
        print(f"🔎 No CSV found. Parsing PDFs in {PDF_DIR} ...")
        pdf_files = glob.glob(os.path.join(PDF_DIR, '*.pdf'))
        if not pdf_files:
            raise FileNotFoundError("❌ No PDF files found in 'pdf_uploads'")

        parsed_frames = [extract_pdf_sales(pdf) for pdf in pdf_files]
        df = pd.concat(parsed_frames, ignore_index=True)

        # Ensure lat/lng columns exist BEFORE saving
        if 'latitude' not in df.columns:
            df['latitude'] = 0.0
        if 'longitude' not in df.columns:
            df['longitude'] = 0.0

        df.to_csv(CSV_PATH, index=False)
        print(f"✅ Parsed and saved data to {CSV_PATH}")
        return df

    else:
        raise FileNotFoundError("❌ CSV and PDF directory both missing. Cannot proceed.")

print("📦 Loading MLS data to PostgreSQL...")
df = get_dataframe()

required_columns = [
    'neighborhood', 'region', 'house_type', 'bedrooms', 'bathrooms', 'sqft',
    'lot_size', 'built_year', 'garage_type', 'address', 'dom',
    'listing_date', 'season',
    'list_price', 'original_price', 'sold_price',
    'latitude', 'longitude'
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in DataFrame: {missing}")

conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Main table
cursor.execute('''
CREATE TABLE IF NOT EXISTS housing_data (
    id SERIAL PRIMARY KEY,
    neighborhood VARCHAR(50),
    region VARCHAR(50),
    house_type VARCHAR(50),
    bedrooms INTEGER,
    bathrooms NUMERIC(3,1),
    sqft INTEGER,
    lot_size NUMERIC(7,2),
    built_year INTEGER,
    garage_type VARCHAR(50),
    address VARCHAR(100),
    dom INTEGER,
    listing_date DATE,
    season VARCHAR(20),
    latitude NUMERIC(9,6),
    longitude NUMERIC(9,6),
    list_price BIGINT,
    original_price BIGINT,
    sold_price BIGINT
);
''')

# Safe alter fallback for future compatibility
column_alterations = [
    ("list_price", "BIGINT"),
    ("original_price", "BIGINT"),
    ("sold_price", "BIGINT"),
    ("latitude", "NUMERIC(9,6)"),
    ("longitude", "NUMERIC(9,6)")
]

for column, col_type in column_alterations:
    cursor.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'housing_data' AND column_name = '{column}'
            ) THEN
                ALTER TABLE housing_data ADD COLUMN {column} {col_type};
            END IF;
        END
        $$;
    """)
# Patch and clip numeric values
numeric_cols = ['bedrooms', 'bathrooms', 'sqft', 'lot_size', 'built_year', 'dom',
                'list_price', 'original_price', 'sold_price']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['list_price'] = df['list_price'].clip(lower=0, upper=9_000_000_000)
df['original_price'] = df['original_price'].clip(lower=0, upper=9_000_000_000)
df['sold_price'] = df['sold_price'].clip(lower=0, upper=9_000_000_000)

insert_data = [tuple(row) for row in df[required_columns].to_numpy()]
insert_sql = f"INSERT INTO housing_data ({', '.join(required_columns)}) VALUES %s"

execute_values(cursor, insert_sql, insert_data)
conn.commit()
cursor.close()
conn.close()
print("✅ Data loaded into PostgreSQL table 'housing_data'")
