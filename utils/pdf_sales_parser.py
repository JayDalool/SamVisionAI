# file: utils/pdf_sales_parser.py

import fitz  # PyMuPDF
import re
import pandas as pd
import datetime
import json
import psycopg2
from typing import Tuple, List

REGION_FILE = "data/region_lookup.json"



def load_region_lookup():
    try:
        with open(REGION_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_region_lookup(region_map):
    with open(REGION_FILE, "w") as f:
        json.dump(region_map, f, indent=2)

def safe_match(pattern, text, cast=str, group=1):
    m = re.search(pattern, text)
    return cast(m.group(group).replace(",", "")) if m else None

def get_season(date):
    return ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"][date.month - 1]

def extract_pdf_sales(pdf_path: str) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    rows = []

    for page in doc:
        text = page.get_text()

        # Field extractors
        list_price = re.search(r"List Price:\s*\$?([\d,]+)", text)
        sold_price = re.search(r"Sell Price:\s*\$?([\d,]+)", text)
        address = re.search(r"\d{3,}\s+.*?Winnipeg.*?(R\d+[A-Z]+\d+)?", text)
        mls_id = re.search(r"MLS.?#:\s*([\d]+)", text)
        dom = re.search(r"DOM[:\s]+(\d+)", text)
        house_type = re.search(r"Type:\s*(\w+)", text)
        bedrooms = re.search(r"Beds[:\s]+(\d+)", text)
        bathrooms = re.search(r"Baths[:\s]+(\d+\.?\d*)", text)
        sqft = re.search(r"Liv Area[:\s]+([\d,]+)", text)
        year_built = re.search(r"Yr Built[:/ ]+(\d{4})", text)
        garage_type = re.search(r"Garage\s*[:\-]?\s*([\w\s\-]+)", text)
        lot_size_match = re.search(r"Lot\s+Dim:\s+(\d+)\s*x\s*(\d+)", text)

        # Derived values
        lot_area = None
        if lot_size_match:
            try:
                lot_area = int(lot_size_match.group(1)) * int(lot_size_match.group(2))
            except:
                lot_area = None

        # Append row
        rows.append({
            "listing_date": datetime.datetime.today().date(),
            "season": get_season(datetime.datetime.today()),
            "address": address.group(0).strip() if address else None,
            "mls_id": mls_id.group(1) if mls_id else None,
            "list_price": int(list_price.group(1).replace(",", "")) if list_price else None,
            "original_price": int(list_price.group(1).replace(",", "")) if list_price else None,
            "sold_price": int(sold_price.group(1).replace(",", "")) if sold_price else None,
            "house_type": house_type.group(1) if house_type else "RD",
            "bedrooms": int(bedrooms.group(1)) if bedrooms else None,
            "bathrooms": float(bathrooms.group(1)) if bathrooms else 0.0,
            "dom": int(dom.group(1)) if dom else None,
            "built_year": int(year_built.group(1)) if year_built else None,
            "garage_type": garage_type.group(1).strip() if garage_type else None,
            "sqft": int(sqft.group(1).replace(",", "")) if sqft else None,
            "lot_size": lot_area,
            "latitude": 0.0,
            "longitude": 0.0,
            "neighborhood": None,
            "region": None
        })

    return pd.DataFrame(rows)


def insert_sales_to_db(df: pd.DataFrame, db_config: dict) -> Tuple[int, List[str]]:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    inserted, skipped = 0, []

    for _, row in df.iterrows():
        try:
            cursor.execute('''
                INSERT INTO housing_data (
                    neighborhood, region, house_type, bedrooms, bathrooms, sqft,
                    lot_size, built_year, garage_type, address, dom,
                    listing_date, season, latitude, longitude,
                    list_price, original_price, sold_price
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                row.get("neighborhood"), row.get("region"), row.get("house_type"),
                row.get("bedrooms"), row.get("bathrooms"), row.get("sqft"),
                row.get("lot_size"), row.get("built_year"), row.get("garage_type"), row.get("address"),
                row.get("dom"), row.get("listing_date"), row.get("season"),
                None, None, row.get("list_price"), row.get("original_price"), row.get("sold_price")
            ))
            inserted += 1
        except:
            skipped.append(row.get("address"))

    conn.commit()
    cursor.close()
    conn.close()
    return inserted, skipped
