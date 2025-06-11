# # file: data/load_to_postgresql.py

# import os
# import sys
# import glob
# import ast
# import pandas as pd
# import psycopg2
# from psycopg2.extras import execute_values
# from utils.pdf_sales_parser import extract_pdf_sales

# CSV_PATH = 'winnipeg_housing_data.csv'
# PDF_DIR = 'pdf_uploads'

# # Load DB config
# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# def parse_bathrooms(bath_str):
#     if isinstance(bath_str, str) and 'F' in bath_str and 'H' in bath_str:
#         parts = bath_str.replace("F", "").replace("H", "/").split("/")
#         try:
#             return int(parts[0]) + 0.5 * int(parts[1])
#         except:
#             return 1.0
#     return 1.0

# def get_dataframe():
#     if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
#         try:
#             df = pd.read_csv(CSV_PATH)

#             # Patch & convert columns
#             df = df.rename(columns={
#                 "Liv Area sqft": "sqft"
#             })
#             df['bathrooms'] = df['bathrooms'].apply(parse_bathrooms)
#             df['built_year'] = pd.to_numeric(df['built_year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce').fillna(0).astype(int)
#             df['listing_date'] = pd.to_datetime('today').date()
#             df['season'] = pd.Timestamp.today().month % 12 // 3
#             df['season'] = df['season'].map({0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'})
#             df['region'] = df['neighborhood'].fillna('none')
#             df['garage_type'] = df['garage_type'].fillna('none')
#             df['original_price'] = df['list_price']

#             # Add missing columns
#             for col in ['latitude', 'longitude']:
#                 if col not in df:
#                     df[col] = 0.0

#             return df
#         except pd.errors.EmptyDataError:
#             print("⚠️ CSV exists but is empty or malformed. Re-parsing from PDFs...")
#             os.remove(CSV_PATH)

#     if os.path.exists(PDF_DIR):
#         print(f"🔎 No CSV found. Parsing PDFs in {PDF_DIR} ...")
#         pdf_files = glob.glob(os.path.join(PDF_DIR, '*.pdf'))
#         if not pdf_files:
#             raise FileNotFoundError("❌ No PDF files found in 'pdf_uploads'")

#         parsed_frames = [extract_pdf_sales(pdf) for pdf in pdf_files]
#         df = pd.concat(parsed_frames, ignore_index=True)

#         # Ensure lat/lng columns exist BEFORE saving
#         if 'latitude' not in df.columns:
#             df['latitude'] = 0.0
#         if 'longitude' not in df.columns:
#             df['longitude'] = 0.0

#         df.to_csv(CSV_PATH, index=False)
#         print(f"✅ Parsed and saved data to {CSV_PATH}")
#         return df

#     else:
#         raise FileNotFoundError("❌ CSV and PDF directory both missing. Cannot proceed.")

# print("[INFO] Loading MLS data to PostgreSQL...")
# df = get_dataframe()

# required_columns = [
#     'neighborhood', 'region', 'house_type', 'bedrooms', 'bathrooms', 'sqft',
#     'lot_size', 'built_year', 'garage_type', 'address', 'dom', 'listing_date',
#     'season', 'latitude', 'longitude', 'list_price', 'original_price', 'sold_price'
# ]

# for col in required_columns:
#     if col not in df:
#         df[col] = 0 if df.dtypes.get(col, '') in [int, float] else 'none'

# conn = psycopg2.connect(**db_config)
# cursor = conn.cursor()

# cursor.execute('''
# CREATE TABLE IF NOT EXISTS housing_data (
#     id SERIAL PRIMARY KEY,
#     neighborhood VARCHAR(50),
#     region VARCHAR(50),
#     house_type VARCHAR(50),
#     bedrooms INTEGER,
#     bathrooms NUMERIC(3,1),
#     sqft INTEGER,
#     lot_size NUMERIC(7,2),
#     built_year INTEGER,
#     garage_type VARCHAR(50),
#     address VARCHAR(100),
#     dom INTEGER,
#     listing_date DATE,
#     season VARCHAR(20),
#     latitude NUMERIC(9,6),
#     longitude NUMERIC(9,6),
#     list_price BIGINT,
#     original_price BIGINT,
#     sold_price BIGINT
# );
# ''')

# numeric_cols = ['bedrooms', 'bathrooms', 'sqft', 'lot_size', 'built_year', 'dom',
#                 'list_price', 'original_price', 'sold_price']
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# df['list_price'] = df['list_price'].clip(0, 9_000_000_000)
# df['original_price'] = df['original_price'].clip(0, 9_000_000_000)
# df['sold_price'] = df['sold_price'].clip(0, 9_000_000_000)

# insert_data = [tuple(row) for row in df[required_columns].to_numpy()]
# insert_sql = f"INSERT INTO housing_data ({', '.join(required_columns)}) VALUES %s"

# execute_values(cursor, insert_sql, insert_data)
# conn.commit()
# cursor.close()
# conn.close()
# print("Data loaded into PostgreSQL table 'housing_data'")
# file: data/load_to_postgresql.py

# file: data/load_to_postgresql.py
# file: data/load_to_postgresql.py
# file: data/load_to_postgresql.py

# import os
# import ast
# import pandas as pd
# import psycopg2
# from utils.pdf_sales_parser import insert_sales_to_db, clean_sales_data, extract_pdf_sales, extract_csv_sales
# from typing import Union, Type

# CSV_OUTPUT = 'winnipeg_housing_data.csv'

# with open("config.txt", "r") as file:
#     db_config = ast.literal_eval(file.read())

# def ensure_numeric(df: pd.DataFrame, col: str, default: Union[int, float] = 0, dtype: Union[Type[int], Type[float]] = int):
#     if col not in df:
#         df[col] = default
#     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default).astype(dtype)

# def load_all_pdf_and_csv(directory: str) -> pd.DataFrame:
#     all_data = []
#     for file in os.listdir(directory):
#         path = os.path.join(directory, file)
#         try:
#             if file.lower().endswith(".pdf"):
#                 print(f"[PDF] Parsing: {file}")
#                 all_data.append(extract_pdf_sales(path))
#             elif file.lower().endswith(".csv"):
#                 print(f"[CSV] Parsing: {file}")
#                 all_data.append(extract_csv_sales(path))
#         except Exception as e:
#             print(f"[ERROR] Failed parsing {file}: {e}")
#     return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# print("[INFO] Loading MLS data from 'pdf_uploads/'...")
# df = load_all_pdf_and_csv("pdf_uploads")

# if df.empty:
#     raise RuntimeError("❌ No valid rows parsed from PDFs or CSVs.")

# df = clean_sales_data(df)
# df.to_csv(CSV_OUTPUT, index=False)

# required_columns = [
#     'neighborhood', 'region', 'house_type', 'bedrooms', 'bathrooms', 'sqft',
#     'lot_size', 'built_year', 'garage_type', 'address', 'dom', 'listing_date',
#     'season', 'latitude', 'longitude', 'list_price', 'original_price', 'sold_price'
# ]

# for col in required_columns:
#     if col not in df:
#         df[col] = 0 if df.dtypes.get(col, '') in [int, float] else 'none'

# conn = psycopg2.connect(**db_config)
# cursor = conn.cursor()

# cursor.execute('''
# CREATE TABLE IF NOT EXISTS housing_data (
#     id SERIAL PRIMARY KEY,
#     neighborhood VARCHAR(50),
#     region VARCHAR(50),
#     house_type VARCHAR(50),
#     bedrooms INTEGER,
#     bathrooms NUMERIC(3,1),
#     sqft INTEGER,
#     lot_size NUMERIC(7,2),
#     built_year INTEGER,
#     garage_type VARCHAR(50),
#     address VARCHAR(100),
#     dom INTEGER,
#     listing_date DATE,
#     season VARCHAR(20),
#     latitude NUMERIC(9,6),
#     longitude NUMERIC(9,6),
#     list_price BIGINT,
#     original_price BIGINT,
#     sold_price BIGINT,
#     UNIQUE(address, listing_date)
# );
# ''')
# conn.commit()
# cursor.close()
# conn.close()

# inserted, skipped = insert_sales_to_db(df, db_config)

# print(f"[OK] Inserted {inserted} records into PostgreSQL.")
# if skipped:
#     print(f"[WARN] Skipped {len(skipped)} row(s). Example: {skipped[:3]}")

# file: data/load_to_postgresql.py

import os
import pandas as pd
import psycopg2
from psycopg2 import sql
from datetime import datetime

CSV_PATH = "parsed_csv/validated.csv"

# Replace with your actual config loading method
from utils.db_config import get_db_config  # expects dict with user, password, host, dbname

def create_table_if_not_exists(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS housing_data (
            id SERIAL PRIMARY KEY,
            neighborhood TEXT,
            region TEXT,
            house_type TEXT,
            bedrooms INTEGER,
            bathrooms NUMERIC(3,1),
            sqft INTEGER,
            lot_size NUMERIC(10,2),
            built_year INTEGER,
            garage_type TEXT,
            address TEXT,
            dom INTEGER,
            listing_date DATE,
            season TEXT,
            latitude NUMERIC(9,6),
            longitude NUMERIC(9,6),
            list_price BIGINT,
            original_price BIGINT,
            sold_price BIGINT
        );
    """)

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise RuntimeError("❌ No data in validated CSV")

    config = get_db_config()
    conn = psycopg2.connect(**config)
    cursor = conn.cursor()

    create_table_if_not_exists(cursor)

    inserted, skipped = 0, []

    for _, row in df.iterrows():
        try:
            values = [
                row.get("neighborhood", "none"),
                row.get("region", "none"),
                row.get("house_type", "none"),
                int(row["bedrooms"]) if pd.notnull(row["bedrooms"]) else 0,
                float(row["bathrooms"]) if pd.notnull(row["bathrooms"]) else 1.0,
                int(row["sqft"]) if pd.notnull(row["sqft"]) else 0,
                float(row["lot_size"]) if pd.notnull(row["lot_size"]) else 0,
                int(row["built_year"]) if pd.notnull(row["built_year"]) else 0,
                row.get("garage_type", "none"),
                row.get("address", "none"),
                int(row["dom"]) if pd.notnull(row["dom"]) else 0,
                row.get("listing_date", datetime.today().date()),
                row.get("season", "none"),
                float(row.get("latitude", 0.0)),
                float(row.get("longitude", 0.0)),
                int(row.get("list_price", 0)),
                int(row.get("original_price", 0)),
                int(row.get("sold_price", 0)),
            ]

            cursor.execute(sql.SQL("""
                INSERT INTO housing_data (
                    neighborhood, region, house_type, bedrooms, bathrooms,
                    sqft, lot_size, built_year, garage_type, address, dom,
                    listing_date, season, latitude, longitude,
                    list_price, original_price, sold_price
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """), values)

            inserted += 1

        except Exception as e:
            skipped.append(str(row.get("address", "unknown")) + " -> " + str(e))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"📥 Inserted {inserted} row(s) into DB.")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} row(s). Example:\n- " + "\n- ".join(skipped[:3]))

if __name__ == "__main__":
    main()

