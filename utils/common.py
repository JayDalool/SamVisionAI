# file: utils/common.py

import re

HOUSE_STYLES = [
    "Bungalow", "Two Storey", "One and a Half", "Split-Level", "Bi-Level",
    "Multi-Level", "Cabin", "Raised Bungalow", "Mobile", "Other"
]

HOUSE_TYPES = [
    "Single Family Detached", "Townhouse", "Condo", "Duplex", "Mobile", "Other"
]

def parse_currency(s: str) -> float:
    return float(s.replace("$", "").replace(",", "")) if s else 0.0

def clean_address_field(address: str) -> str:
    return re.sub(r"\s+", " ", address.strip())

def normalize_garage_type(raw: str) -> str:
    return raw.lower().replace("\n", ", ").strip() if raw else "none"

def match_known_terms(text: str, options: list[str]) -> str:
    found = [opt for opt in options if opt.lower() in text.lower()]
    return ", ".join(found) if found else "Other"

def parse_bedrooms(text: str) -> int:
    match = re.search(r"(\d+)\s*BR", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def parse_bathrooms(text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*Bath", text, re.IGNORECASE)
    return float(match.group(1)) if match else 1.0

def get_current_season() -> str:
    from datetime import datetime
    month = datetime.today().month
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"
