
# file: utils/pdf_sales_parser.py

import re
import pandas as pd
from datetime import datetime
from typing import List
from pdfminer.high_level import extract_text
import os


def extract_text_from_path(path: str) -> List[str]:
    if path.lower().endswith(".pdf"):
        text = extract_text(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    return text.split("Page ")


def get_current_season():
    return ["Winter", "Spring", "Summer", "Fall"][(datetime.today().month % 12) // 3]


def parse_currency(value: str) -> float:
    try:
        return float(value.replace("$", "").replace(",", ""))
    except:
        return 0.0


def parse_bedrooms(text: str) -> int:
    match = re.search(r"BDA:\s*(\d+)", text)
    return int(match.group(1)) if match else 3


def parse_bathrooms(text: str) -> float:
    fb = re.search(r"FB:\s*(\d+)", text)
    hb = re.search(r"HB:\s*(\d+)", text)
    return float(fb.group(1)) + 0.5 * float(hb.group(1)) if fb and hb else 2.0


def normalize_garage_type(raw: str) -> str:
    raw = raw.lower().strip() if raw else ""
    if not raw or raw == 'none':
        return "none"
    tags = []
    if "pad" in raw:
        tags.append("Parking pad")
    if "plug" in raw:
        tags.append("plug-in")
    if "rear" in raw:
        tags.append("rear drive")
    if "unpaved" in raw:
        tags.append("unpaved")
    if "attached" in raw:
        tags.append("attached")
    if "detached" in raw:
        tags.append("detached")
    if "single" in raw:
        tags.append("single")
    if "double" in raw:
        tags.append("double")
    if "carport" in raw:
        tags.append("carport")
    return ", ".join(sorted(set(tags))) if tags else raw


def clean_address_field(addr: str) -> str:
    addr = re.sub(r"\n", " ", addr)
    addr = re.sub(r"Winnipeg Regional.*?Levies", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r"\d+\s+Sold", "", addr)
    addr = re.sub(r"\s+", " ", addr).strip()
    return addr


def extract_pdf_sales(path: str) -> pd.DataFrame:
    pages = extract_text_from_path(path)
    records = []

    known_neighborhoods = [
        "Riverview", "Lord Roberts", "Fort Rouge", "Osborne Village", "Crescentwood",
        "Charleswood", "Richmond West", "Fairfield Park", "Deer Pointe", "Headingley South",
        "St Boniface", "Norwood", "Norwood Flats", "East Fort Garry", "Wildwood",
        "East Kildonan", "St Vital", "St Norbert", "St James", "Tuxedo",
        "River Heights", "Linden Woods", "Garden City", "Transcona", "West End"
    ]

    for page_text in pages:
        entries = re.split(r"(?=Sold\d{8})", page_text)
        for listing in entries:
            if not listing.strip():
                continue

            address_match = re.search(r"(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", listing)
            mls_match = re.search(r"MLS[\u00aeR#\s:]*?(\d{8})", listing)
            dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
            sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
            lot_match = re.search(r"(\d+\.\d+)\s*M2", listing)
            price_matches = re.findall(r"\$(\d{1,3}(?:,\d{3})*)", listing)

            raw_addr = address_match.group(1).strip() if address_match else f"Unknown Address"
            address = clean_address_field(raw_addr)
            address_lower = address.lower()
            listing_lower = listing.lower()

            neighborhood = "Loose Area"
            for hood in known_neighborhoods:
                hood_lower = hood.lower()
                if hood_lower in listing_lower or hood_lower in address_lower:
                    neighborhood = hood
                    break

            if neighborhood == "Loose Area":
                suffix_map = {
                    "beresford": "Lord Roberts",
                    "arnold": "Riverview",
                    "wardlaw": "Osborne Village",
                    "mcmillan": "Crescentwood",
                    "fleet": "Osborne Village",
                    "walker": "Lord Roberts",
                    "ashland": "Riverview",
                    "balfour": "Riverview",
                    "montgomery": "Riverview",
                    "morley": "Lord Roberts",
                    "clark": "Riverview",
                    "baltimore": "Riverview",
                    "churchill": "Fort Rouge",
                    "dudley": "Crescentwood",
                    "mulvey": "Crescentwood",
                    "scotland": "Crescentwood",
                    "warsaw": "Crescentwood",
                    "corydon": "Crescentwood",
                    "hector": "Crescentwood",
                    "gauvin": "St Boniface",
                    "archibald": "St Boniface",
                    "deniset": "St Boniface",
                    "lariviere": "St Boniface",
                    "lyndale": "Norwood Flats",
                    "eugenie": "Norwood Flats",
                    "redview": "St Vital",
                    "kitson": "St Boniface",
                    "dollard": "St Boniface",
                    "langevin": "St Boniface"
                }
                for suffix, hood in suffix_map.items():
                    if suffix in address_lower:
                        neighborhood = hood
                        break

            mls_number = mls_match.group(1) if mls_match else "unknown"
            dom = int(dom_match.group(1)) if dom_match else 0
            sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200

            lot_size = 0.0
            if lot_match:
                lot_val = lot_match.group(1)
                try:
                    lot_size = float(lot_val)
                except:
                    lot_size = 0.0

            list_price = parse_currency(price_matches[-4]) if len(price_matches) >= 4 else 0.0
            sold_price = parse_currency(price_matches[-2]) if len(price_matches) >= 4 else 0.0

            garage_match = re.search(r"Parking[\s\S]{0,100}?((?:Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", listing, re.IGNORECASE)
            garage_raw = garage_match.group(1) if garage_match else ""
            garage_type = normalize_garage_type(garage_raw)

            house_type_match = re.search(r"Property Type\s*:?.*?(Single Family Detached|Townhouse|Bungalow|Duplex|Condo|Mobile|Other)", listing, re.IGNORECASE)
            house_type = house_type_match.group(1).strip() if house_type_match else "Single Family Detached"

            bedrooms = parse_bedrooms(listing)
            bathrooms = parse_bathrooms(listing)

            records.append({
                "listing_date": datetime.today().date(),
                "season": get_current_season(),
                "mls_number": mls_number,
                "neighborhood": neighborhood,
                "address": address,
                "list_price": list_price,
                "sold_price": sold_price,
                "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
                "dom": dom,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "garage_type": garage_type,
                "house_type": house_type,
                "sqft": sqft,
                "lot_size": lot_size,
            })

    return pd.DataFrame(records)
