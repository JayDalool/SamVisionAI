# file: utils/txt_sales_parser.py
import os
import re
import hashlib
import pandas as pd
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import (
    clean_address_field, parse_currency, normalize_garage_type,
    match_known_terms, parse_bedrooms, parse_bathrooms,
    get_current_season, HOUSE_STYLES, HOUSE_TYPES
)

class TxtSalesParser:
    def __init__(self, known_neighborhoods, suffix_map):
        self.known_neighborhoods = known_neighborhoods
        self.suffix_map = suffix_map
        self.address_regex = re.compile(
            r"\b\d{1,5}(?:[-\d]*)?\s+[A-Z][a-zA-Z0-9\s\.'\-]*\s+(?:Ave(?:nue)?|St(?:reet)?|Dr(?:ive)?|Rd(?:oad)?|Blvd|Boulevard|Ln|Lane|Bay|Crescent|Pl(?:ace)?)\b",
            re.IGNORECASE
        )

    def extract_price_by_label(self, label: str, text: str) -> float:
        match = re.search(rf"{label}\s*\$?([\d,]+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
        return 0.0

    def parse_folder(self, folder_path: str) -> pd.DataFrame:
        dfs = []
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                txt_path = os.path.join(folder_path, file)
                print(f"Parsing {txt_path}")
                df = self.parse_file(txt_path)
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def normalize_basement(self, basement_raw: str) -> str:
        val = basement_raw.strip().lower()
        if "none" in val:
            return "No Basement"
        elif "crawl" in val:
            return "Crawl Space"
        elif "3/4" in val:
            return "Three-Quarter"
        elif "full" in val:
            return "Full"
        return basement_raw.strip().title()

    def parse_file(self, path: str) -> pd.DataFrame:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        listings = [(m.start(), m.group()) for m in self.address_regex.finditer(text)]
        records = []

        for i, (start, _) in enumerate(listings):
            end = listings[i + 1][0] if i + 1 < len(listings) else len(text)
            block = text[start:end]

            address_match = self.address_regex.search(block)
            raw_address = address_match.group().strip() if address_match else "Unknown Address"
            address = clean_address_field(raw_address)
            address_lower = address.lower()
            listing_lower = block.lower()

            if any(x in address_lower for x in [" list", " winnipeg", " regional"]):
                continue  # Skip invalid lines

            neighborhood = next((hood for hood in self.known_neighborhoods if hood.lower() in listing_lower or hood.lower() in address_lower), "Loose Area")
            if neighborhood == "Loose Area":
                for suffix, hood in self.suffix_map.items():
                    if suffix in address_lower:
                        neighborhood = hood
                        break

            mls = re.search(r"MLS[\u00aeR#:\s]*?(\d{8})", block)
            mls_number = mls.group(1) if mls else hashlib.md5(address.encode()).hexdigest()[:10]

            dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", block, re.IGNORECASE)
            dom = int(dom_match.group(1)) if dom_match else 0
            sqft_match = re.search(r"(\d{3,5})\s+SF", block)
            sqft = int(sqft_match.group(1)) if sqft_match else 1200
            lot_size_match = re.search(r"(\d+\.\d+)\s*M2", block)
            lot_size = float(lot_size_match.group(1)) if lot_size_match else 0.0

            list_price = self.extract_price_by_label("List Price", block)
            sold_price = self.extract_price_by_label("Sold Price", block)

            garage_match = re.search(r"Parking[\s\S]{0,100}?((Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", block, re.IGNORECASE)
            garage_type = normalize_garage_type(garage_match.group(1)) if garage_match else "none"

            basement_match = re.search(r"Basement\s*[:\-]?\s*([A-Za-z0-9 /]+)", block, re.IGNORECASE)
            basement = self.normalize_basement(basement_match.group(1)) if basement_match else "Unknown"

            style = match_known_terms(block, HOUSE_STYLES)
            type_ = match_known_terms(block, HOUSE_TYPES)
            if "Other" in type_ and len(type_.split(",")) > 1:
                type_ = ", ".join(t for t in type_.split(",") if t.strip() != "Other")

            house_type_match = re.search(r"Property Type\s*:?.*?(Single Family Detached|Townhouse|Bungalow|Duplex|Condo|Mobile|Other)", block, re.IGNORECASE)
            house_type = house_type_match.group(1).strip() if house_type_match else "Single Family Detached"

            bedrooms = parse_bedrooms(block)
            bathrooms = parse_bathrooms(block)

            records.append({
                "listing_date": datetime.today().date(),
                "season": get_current_season(),
                "mls_number": mls_number,
                "neighborhood": neighborhood,
                "address": address,
                "list_price": list_price,
                "sold_price": sold_price,
                "sell_list_ratio": round(sold_price / list_price, 2) if list_price else 0.0,
                "dom": dom,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "garage_type": garage_type,
                "basement": basement,
                "house_type": house_type,
                "style": style,
                "type": type_,
                "sqft": sqft,
                "lot_size": lot_size,
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    known_neighborhoods = [
        "Riverview", "Lord Roberts", "Fort Rouge", "Osborne Village", "Crescentwood",
        "Charleswood", "Richmond West", "Fairfield Park", "Deer Pointe", "Headingley South",
        "St Boniface", "Norwood", "Norwood Flats", "East Fort Garry", "Wildwood",
        "East Kildonan", "St Vital", "St Norbert", "St James", "Tuxedo",
        "River Heights", "Linden Woods", "Garden City", "Transcona", "West End"
    ]

    suffix_map = {
        "beresford": "Lord Roberts", "arnold": "Riverview", "wardlaw": "Osborne Village", "mcmillan": "Crescentwood",
        "fleet": "Osborne Village", "walker": "Lord Roberts", "ashland": "Riverview", "balfour": "Riverview",
        "montgomery": "Riverview", "morley": "Lord Roberts", "clark": "Riverview", "baltimore": "Riverview",
        "churchill": "Fort Rouge", "dudley": "Crescentwood", "mulvey": "Crescentwood", "scotland": "Crescentwood",
        "warsaw": "Crescentwood", "corydon": "Crescentwood", "hector": "Crescentwood", "gauvin": "St Boniface",
        "archibald": "St Boniface", "deniset": "St Boniface", "lariviere": "St Boniface", "lyndale": "Norwood Flats",
        "eugenie": "Norwood Flats", "redview": "St Vital", "kitson": "St Boniface", "dollard": "St Boniface",
        "langevin": "St Boniface"
    }

    parser = TxtSalesParser(known_neighborhoods, suffix_map)
    output_df = parser.parse_folder("pdf_uploads")
    output_df.to_csv("parsed_csv/from_txt_parser.csv", index=False)
    print("✅ Saved parsed CSV to parsed_csv/from_txt_parser.csv")
