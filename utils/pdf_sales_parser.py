# file: utils/pdf_sales_parser.py

import fitz
import re
import pandas as pd
import datetime
import json
from typing import Tuple, List, Callable, Any, Union
import psycopg2

REGION_FILE = "data/region_lookup.json"

def get_season(date):
    return ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"][date.month - 1]

def safe_match(pattern: str, text: str, cast: Callable[[Any], Any] = str, group: Union[int, Tuple[int, ...]] = 1) -> Any:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        if isinstance(group, tuple):
            values = [match.group(g).replace(",", "").strip() for g in group]
            return cast(values)
        value = match.group(group).replace(",", "").strip()
        return cast(value)
    except:
        return None

def normalize_garage_type(raw: str) -> str:
    raw = str(raw).lower()
    mapping = {
        "single_attached": ["single attached", "sgl att"],
        "single_detached": ["single detached", "sgl det"],
        "double_attached": ["double attached", "dbl att"],
        "double_detached": ["double detached", "dbl det"],
        "triple_attached": ["triple attached", "tpl att"],
        "triple_detached": ["triple detached", "tpl det"],
    }
    for key, vals in mapping.items():
        if any(v in raw for v in vals):
            return key
    return "none"

def extract_pdf_sales(pdf_path: str) -> pd.DataFrame:
    import fitz
    doc = fitz.open(pdf_path)
    rows = []

    for page in doc:
        text = page.get_text("text")  # better sequential layout
        lot_area_m2 = safe_match(r"Lot Area:\s*([\d,.]+)\s*M2", text, float)
        lot_front = safe_match(r"Lot Front:\s*([\d.]+)\s*M", text, float)
        lot_depth = safe_match(r"Lot Dpth:\s*([\d.]+)\s*M", text, float)

        lot_area_sqft = None
        if lot_area_m2:
            lot_area_sqft = round(lot_area_m2 * 10.7639)
        elif lot_front and lot_depth:
            lot_area_sqft = round(lot_front * lot_depth * 10.7639)

        address_match = re.search(r"\d{2,5} [^\n,]+(?:,|\s)Winnipeg", text)
        address = address_match.group(0).strip() if address_match else None

        data = {
            "listing_date": datetime.date.today(),
            "season": get_season(datetime.date.today()),
            "address": address,
            "list_price": safe_match(r"List Price:\s*\$?([\d,]+)", text, int),
            "original_price": safe_match(r"Original Price:\s*\$?([\d,]+)", text, int),
            "sold_price": safe_match(r"(Sell Price|Selling Price):\s*\$?([\d,]+)", text, int, group=2),
            "house_type": safe_match(r"Type:\s*([^\n]+)", text),
            "bedrooms": safe_match(r"BDA:\s*(\d+)", text, int),
            "bathrooms": safe_match(r"FB:\s*(\d+).*HB:\s*(\d+)", text, lambda x: int(x[0]) + 0.5 * int(x[1]), group=(0, 1)),
            "dom": safe_match(r"DOM[:\s]+(\d+)", text, int),
            "built_year": safe_match(r"/\s*(\d{4})", text, int),
            "garage_type": normalize_garage_type(safe_match(r"Parking:\s*([^\n]*)", text)),
            "sqft": safe_match(r"(\d{3,5})\s*SF", text, int),
            "lot_size": lot_area_sqft,
            "latitude": 0.0,
            "longitude": 0.0,
            "neighborhood": safe_match(r"\d{7}\s+([^\n]+)", text),
            "region": safe_match(r"\d{7}\s+([^\n]+)", text),
        }

        # skip row if critical fields missing
        if not data["address"] or not data["sold_price"]:
            print(f"[SKIP] Missing address or sold_price in: {pdf_path}")
            continue

        rows.append(data)

    return clean_sales_data(pd.DataFrame(rows))


def extract_csv_sales(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Liv Area sqft": "sqft"})
    if df["bathrooms"].dtype == object:
        df["bathrooms"] = df["bathrooms"].str.extract(r"F(\d+)/H(\d+)").fillna(0).astype(int).sum(axis=1)
    df["built_year"] = df["built_year"].astype(str).str.extract(r"(\d{4})").astype(float).fillna(0).astype(int)
    df["listing_date"] = datetime.datetime.today().date()
    df["season"] = get_season(datetime.datetime.today())
    df["region"] = df["neighborhood"].fillna("none")
    df["garage_type"] = df["garage_type"].apply(lambda x: normalize_garage_type(str(x)))
    df["original_price"] = pd.to_numeric(df["list_price"].replace(r"[\$,]", "", regex=True), errors='coerce').fillna(0)
    df["list_price"] = pd.to_numeric(df["list_price"].replace(r"[\$,]", "", regex=True), errors='coerce').fillna(0)
    df["sold_price"] = pd.to_numeric(df["sold_price"].replace(r"[\$,()]", "", regex=True), errors='coerce').fillna(0)
    return clean_sales_data(df)

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        with open(REGION_FILE) as f:
            region_map = json.load(f)
    except:
        region_map = {}

    for i, row in df.iterrows():
        if not row.get("season"):
            df.at[i, "season"] = get_season(row.get("listing_date", datetime.datetime.today()))

        if isinstance(row.get("garage_type"), str):
            val = row["garage_type"].lower().replace("dbl", "double").replace("sgl", "single")
            df.at[i, "garage_type"] = val.strip()

        if row.get("sqft") and (row["sqft"] <= 0 or row["sqft"] > 10000):
            df.at[i, "sqft"] = 0

        if row.get("lot_size") and (row["lot_size"] <= 0 or row["lot_size"] > 100000):
            df.at[i, "lot_size"] = 0

        if isinstance(row.get("address"), str):
            df.at[i, "address"] = row["address"].strip()
            for key in region_map:
                if key.lower() in row["address"].lower():
                    df.at[i, "region"] = region_map[key]
                    break

    df["sold_price"] = df["sold_price"].fillna(df["list_price"])
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors='coerce').fillna(1.0)
    for col in ["garage_type", "region", "neighborhood"]:
        df[col] = df[col].fillna("none").astype(str)

    return df

def insert_sales_to_db(df: pd.DataFrame, db_config: dict) -> tuple[int, list[str]]:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    inserted, skipped = 0, []

    required_columns = [
        'neighborhood', 'region', 'house_type', 'bedrooms', 'bathrooms', 'sqft',
        'lot_size', 'built_year', 'garage_type', 'address', 'dom',
        'listing_date', 'season', 'latitude', 'longitude',
        'list_price', 'original_price', 'sold_price'
    ]

    for _, row in df.iterrows():
        try:
            values = [
                None if pd.isnull(row[col]) else
                float(row[col]) if col in ['bathrooms', 'lot_size', 'latitude', 'longitude'] else
                int(row[col]) if col in ['bedrooms', 'sqft', 'built_year', 'dom', 'list_price', 'original_price', 'sold_price'] else
                str(row[col]) for col in required_columns
            ]

            insert_sql = f"""
                INSERT INTO housing_data ({', '.join(required_columns)})
                VALUES ({', '.join(['%s'] * len(required_columns))})
                ON CONFLICT (address, listing_date) DO NOTHING
            """
            cursor.execute(insert_sql, values)
            conn.commit()
            inserted += 1
        except Exception as e:
            conn.rollback()  
            print(f"[SKIP] Row failed: {row.get('address', 'no_address')} -> {e}")
            skipped.append(str(row.get('address', 'no_address')))

    cursor.close()
    conn.close()
    return inserted, skipped
