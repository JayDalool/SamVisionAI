# # file: data/load_to_postgresql.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import psycopg2
from psycopg2 import sql
from datetime import datetime
from utils.db_config import get_db_config

CSV_PATH = "parsed_csv/validated.csv"


def safe_str(val, default="none", maxlen=255):
    try:
        return str(val)[:maxlen] if pd.notnull(val) else default
    except:
        return default
    
def ensure_database_exists(base_config, dbname="SamVision"):
    config_no_db = base_config.copy()
    config_no_db.pop("dbname", None)

    dsn = (
        f"dbname=postgres "
        f"user={config_no_db['user']} "
        f"password={config_no_db['password']} "
        f"host={config_no_db['host']} "
        f"port={config_no_db['port']}"
    )
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
    exists = cur.fetchone()
    if not exists:
        print(f"📦 Creating database '{dbname}'...")
        cur.execute(sql.SQL('CREATE DATABASE "{}"').format(sql.Identifier(dbname)))
    else:
        print(f"✅ Database '{dbname}' already exists.")

    cur.close()
    conn.close()


def ensure_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "mls_number": "none",
        "neighborhood": "none",
        "region": "none",
        "house_type": "none",
        "bedrooms": 0,
        "bathrooms": 1.0,
        "sqft": 0,
        "lot_size": 0.0,
        "built_year": 0,
        "age": 0,
        "garage_type": "none",
        "address": "none",
        "dom_days": 0,                     
        "basement_type": "none",           
        "listing_date": datetime.today().date(),
        "season": "Summer",
        "latitude": 0.0,
        "longitude": 0.0,
        "list_price": 0,
        "sold_price": 0,
        "sell_list_ratio": 1.0,
        "style": "none",
        "type": "none"
    }

    for col, default in defaults.items():
        if col not in df:
            df[col] = default
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
            else:
                df[col] = df[col].fillna(default)
    return df

def create_table_if_not_exists(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS housing_data (
            id SERIAL PRIMARY KEY,
            mls_number TEXT,
            neighborhood TEXT,
            region TEXT,
            house_type TEXT,
            bedrooms INTEGER,
            bathrooms NUMERIC(3,1),
            sqft INTEGER,
            lot_size NUMERIC(10,2),
            built_year INTEGER,
            age INTEGER,
            garage_type TEXT,
            address TEXT,
            dom_days INTEGER,                       
            basement_type TEXT,                     
            listing_date DATE,
            season TEXT,
            latitude NUMERIC(9,6),
            longitude NUMERIC(9,6),
            list_price BIGINT,
            sold_price BIGINT,
            sell_list_ratio NUMERIC(5,2),
            style TEXT,
            type TEXT,
            UNIQUE(address, listing_date)
        );
    """)

    # Ensure columns exist even if table already created
    cursor.execute("ALTER TABLE housing_data ADD COLUMN IF NOT EXISTS dom_days INTEGER;")
    cursor.execute("ALTER TABLE housing_data ADD COLUMN IF NOT EXISTS basement_type TEXT;")

def main():
    print(f"🚀 Starting DB Load | {datetime.now().isoformat()}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise RuntimeError("❌ No data in validated CSV")

    df = ensure_all_columns(df)

    numeric_cols = ["bedrooms", "sqft", "built_year", "age", "dom_days", "list_price", "sold_price"]
    float_cols = ["bathrooms", "lot_size", "latitude", "longitude", "sell_list_ratio"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce").dt.date
    df = df[df["listing_date"].notnull()]

    # 🔑 1) Base config (from your config / get_db_config)
    base_config = get_db_config()

    # 🔑 2) Ensure DB exists (using same user/pass/host/port)
    ensure_database_exists(base_config, dbname=base_config.get("dbname", "SamVision"))
    dsn = (
        f"dbname={base_config['dbname']} "
        f"user={base_config['user']} "
        f"password={base_config['password']} "
        f"host={base_config['host']} "
        f"port={base_config['port']}"
    )
    conn = psycopg2.connect(dsn)
    cursor = conn.cursor()



    create_table_if_not_exists(cursor)

    inserted, skipped = 0, []

    for _, row in df.iterrows():
        try:
            if not row.get("address") or pd.isna(row.get("address")):
                raise ValueError("Missing address")
            if not row.get("listing_date") or pd.isna(row.get("listing_date")):
                raise ValueError("Missing listing_date")

            values = [
                safe_str(row["mls_number"], maxlen=50),
                safe_str(row["neighborhood"], maxlen=100),
                safe_str(row["region"], maxlen=100),
                safe_str(row["house_type"], maxlen=100),
                int(row["bedrooms"]),
                float(row["bathrooms"]),
                int(row["sqft"]),
                float(row["lot_size"]),
                int(row["built_year"]),
                int(row["age"]),
                safe_str(row["garage_type"], maxlen=100),
                safe_str(row["address"], maxlen=100),
                int(row["dom_days"]),
                safe_str(row["basement_type"], maxlen=50),
                row["listing_date"],
                safe_str(row["season"], maxlen=50),
                float(row["latitude"]),
                float(row["longitude"]),
                int(row["list_price"]),
                int(row["sold_price"]),
                float(row["sell_list_ratio"]),
                safe_str(row["style"], maxlen=100),
                safe_str(row["type"], maxlen=100)
            ]

            cursor.execute(sql.SQL("""
                INSERT INTO housing_data (
                    mls_number, neighborhood, region, house_type, bedrooms, bathrooms,
                    sqft, lot_size, built_year, age, garage_type, address,
                    dom_days, basement_type, listing_date, season,
                    latitude, longitude, list_price, sold_price,
                    sell_list_ratio, style, type
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (address, listing_date) DO NOTHING
            """), values)
            inserted += 1

        except Exception as e:
            skipped.append(f"{row.get('address', 'unknown')} -> {e}")

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM housing_data;")
    fetch_result = cursor.fetchone()
    total_rows = fetch_result[0] if fetch_result is not None else 0
    cursor.close()
    conn.close()

    print(f"✅ Inserted {inserted} row(s) into DB.")
    print(f"✅ DB now contains {total_rows} total row(s) in housing_data.")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} row(s). Example:\n- " + "\n- ".join(skipped[:5]))


if __name__ == "__main__":
    main()
