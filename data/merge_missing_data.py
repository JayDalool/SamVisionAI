import pandas as pd
import os
import re
from datetime import datetime

CSV_PATH = "data/winnipeg_housing_data.csv"
MISSING_PATH = "data/missing_addresses.txt"

FIELDS = [
    "listing_date", "season", "address", "list_price", "original_price", "sold_price",
    "house_type", "bedrooms", "bathrooms", "dom", "built_year", "garage_type",
    "sqft", "lot_size", "latitude", "longitude", "neighborhood", "region"
]

def parse_missing_txt(text):
    listings = []
    records = re.split(r"(?=\d+\s+.+?\s+Winnipeg\s+R\d[A-Z]\d\s\d[A-Z]\d)", text)
    for record in records:
        if not record.strip():
            continue

        addr = re.search(r"^(\d+\s.+?)\s+Winnipeg", record)
        nghbrhd = re.search(r"Nghbrhd:\s*(.+)", record)
        price = re.search(r"List Price:\s*\$([\d,]+)", record)
        sold = re.search(r"Sell\s*Price:\s*\$([\d,]+)", record)
        date = re.search(r"Sell\s*Date:\s*(\d{2}/\d{2}/\d{4})", record)
        dom = re.search(r"DOM:\s*(\d+)", record)
        year = re.search(r"Yr Built.*?:\s*(\d{4})", record)
        sqft = re.search(r"Liv Area:\s*[\d.,]+\s*M2.*?(\d[\d,]+)\s*SF", record)
        garage = re.search(r"Parking:\s*(.+)", record)
        house_type = re.search(r"Type:\s*(\w+)", record)
        beds = re.search(r"BDA:\s*(\d+)", record)
        baths = re.search(r"Baths:.*?(\d)\s*/?H?(\d)?", record)

        listings.append({
            "listing_date": datetime.today().strftime("%Y-%m-%d"),
            "season": "Spring",
            "address": addr.group(1).strip() + " Winnipeg" if addr else None,
            "list_price": int(price.group(1).replace(",", "")) if price else None,
            "original_price": int(price.group(1).replace(",", "")) if price else None,
            "sold_price": int(sold.group(1).replace(",", "")) if sold else None,
            "house_type": house_type.group(1).strip() if house_type else None,
            "bedrooms": int(beds.group(1)) if beds else None,
            "bathrooms": int(baths.group(1)) + int(baths.group(2)) * 0.5 if baths and baths.group(2) else int(baths.group(1)) if baths else None,
            "dom": int(dom.group(1)) if dom else None,
            "built_year": int(year.group(1)) if year else None,
            "garage_type": garage.group(1).strip().lower() if garage else None,
            "sqft": int(sqft.group(1).replace(",", "")) if sqft else None,
            "lot_size": None,
            "latitude": 0.0,
            "longitude": 0.0,
            "neighborhood": nghbrhd.group(1).strip() if nghbrhd else None,
            "region": nghbrhd.group(1).strip() if nghbrhd else None,
        })
    return listings

def main():
    if os.path.exists(CSV_PATH):
        try:
            df_existing = pd.read_csv(CSV_PATH)
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame(columns=FIELDS)
    else:
        df_existing = pd.DataFrame(columns=FIELDS)

    with open(MISSING_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    parsed = parse_missing_txt(raw)
    df_new = pd.DataFrame(parsed)

    if not df_new.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["address", "sold_price"], inplace=True)
        df_combined.to_csv(CSV_PATH, index=False)
        print(f"Combined dataset saved to {CSV_PATH}. {len(df_new)} new records added.")
    else:
        print("⚠️ No valid records found in missing_addresses.txt")

if __name__ == "__main__":
    main()
