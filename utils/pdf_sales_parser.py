# file: utils/pdf_sales_parser.py

import fitz
import re
import pandas as pd
import datetime
import json
from typing import Tuple, List, Callable, Any, Union

REGION_FILE = "data/region_lookup.json"
MISSING_ADDR_LOG = "data/missing_addresses.txt"

def get_season(date):
    return ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"][date.month - 1]

def safe_match(pattern: str, text: str, cast: Callable[[Any], Any] = str, group: Union[int, Tuple[int, ...]] = 1) -> Any:
    match = re.search(pattern, text)
    if not match:
        return None
    if isinstance(group, tuple):
        try:
            values = [match.group(g).replace(",", "").strip() for g in group]
            return cast(values)
        except:
            return None
    else:
        try:
            value = match.group(group).replace(",", "").strip()
            return cast(value)
        except:
            return None

def normalize_garage_type(raw: str) -> str:
    raw = raw.lower().strip()
    if "single" in raw and "attached" in raw:
        return "single_attached"
    if "single" in raw and "detached" in raw:
        return "single_detached"
    if "double" in raw and "attached" in raw:
        return "double_attached"
    if "double" in raw and "detached" in raw:
        return "double_detached"
    if "triple" in raw and "attached" in raw:
        return "triple_attached"
    if "triple" in raw and "detached" in raw:
        return "triple_detached"
    return "none"

def extract_pdf_sales(pdf_path: str) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    rows = []

    for page in doc:
        text = page.get_text()

        lot_area_m2 = safe_match(r"Lot Area:\s*([\d,.]+)\s*M2", text, lambda x: float(x))
        lot_front = safe_match(r"Lot Front:\s*([\d.]+)\s*M", text, float)
        lot_depth = safe_match(r"Lot Dpth:\s*([\d.]+)\s*M", text, float)

        lot_area_sqft = None
        if lot_area_m2:
            lot_area_sqft = round(lot_area_m2 * 10.7639)
        elif lot_front and lot_depth:
            lot_area_sqft = round(lot_front * lot_depth * 10.7639)

        # Improved address extraction: Greedy lines starting with number and capitalized words
        lines = text.split("\n")
        address = None
        for line in lines:
            if re.match(r"^\d{3,5} [\w\s,.#'-]+Winnipeg", line):
                address = line.strip()
                break

        data = {
            "listing_date": datetime.datetime.today().date(),
            "season": get_season(datetime.datetime.today()),
            "address": address,
            "list_price": safe_match(r"List Price:\s*\$?([\d,]+)", text, int),
            "original_price": safe_match(r"List Price:\s*\$?([\d,]+)", text, int),
            "sold_price": safe_match(r"Sell Price:\s*\$?([\d,]+)", text, int),
            "house_type": safe_match(r"Type:\s*(\w+)", text),
            "bedrooms": safe_match(r"BDA:\s*(\d+)", text, int),
            "bathrooms": safe_match(r"Baths:\s*F(\d+)/H(\d+)", text, lambda x: int(x[0]) + int(x[1]), group=(1, 2)),
            "dom": safe_match(r"DOM[:\s]+(\d+)", text, int),
            "built_year": safe_match(r"Yr Built(?:/Age)?:\s*(\d{4})", text, int),
            "garage_type": normalize_garage_type(safe_match(r"Parking:\s*([^\n]*)", text) or "none"),
            "sqft": safe_match(r"Liv Area:\s*([\d.]+)\s*M2", text, lambda x: round(float(x) * 10.7639)),
            "lot_size": lot_area_sqft,
            "latitude": 0.0,
            "longitude": 0.0,
            "neighborhood": safe_match(r"Nghbrhd:\s*([^\n]+)", text),
            "region": safe_match(r"Nghbrhd:\s*([^\n]+)", text)
        }

        rows.append(data)

    df = pd.DataFrame(rows)
    return clean_sales_data(df)

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        with open(REGION_FILE) as f:
            region_map = json.load(f)
    except:
        region_map = {}

    for i, row in df.iterrows():
        if not row["season"]:
            df.at[i, "season"] = get_season(row["listing_date"])

        if not row["garage_type"]:
            df.at[i, "garage_type"] = "none"

        if row["garage_type"]:
            val = row["garage_type"].lower()
            val = val.replace("dbl", "double").replace("sgl", "single").replace("att", "attached").replace("det", "detached")
            df.at[i, "garage_type"] = val.strip()

        if row["sqft"] is not None and (row["sqft"] <= 0 or row["sqft"] > 10000):
            df.at[i, "sqft"] = None

        if row["lot_size"] is not None and (row["lot_size"] <= 0 or row["lot_size"] > 100000):
            df.at[i, "lot_size"] = None

        if isinstance(row["address"], str):
            df.at[i, "address"] = row["address"].strip()
            for key in region_map:
                if key.lower() in row["address"].lower():
                    df.at[i, "region"] = region_map[key]
                    break

    df["sold_price"] = df["sold_price"].fillna(df["list_price"])
    df["bathrooms"] = df["bathrooms"].fillna(1)

    for col in ["garage_type", "region", "neighborhood"]:
        df[col] = df[col].fillna("none")

    return df

if __name__ == "__main__":
    import sys
    from pprint import pprint

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "pdf_uploads/sample.pdf"
    df = extract_pdf_sales(pdf_path)
    pprint(df.head())