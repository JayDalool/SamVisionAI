
# file: utils/pdf_sales_parser.py

import re, hashlib, pandas as pd, json, os, time, random
from datetime import datetime, timedelta
from pdfminer.high_level import extract_text
import requests
from typing import TYPE_CHECKING

# --- Configurations ---
HOUSE_STYLES = ["Bungalow", "One and a Half", "Two Storey", "Bi-Level", "Split-Level", "Cab-Over", "Multi-Level", "Three Storey", "Back Split", "Side Split"]
HOUSE_TYPES = ["Single Family Detached", "Single Family Attached", "Condo", "Townhouse", "Duplex", "Triplex", "Fourplex", "Mobile Home", "Other"]

try:
    from rapidfuzz import process, fuzz
    USE_FUZZY = True
except ImportError:
    from types import SimpleNamespace
    process = SimpleNamespace(extractOne=lambda *a, **k: None)
    fuzz = SimpleNamespace(token_sort_ratio=None)
    USE_FUZZY = False

NEIGHBORHOOD_ALIASES = {
    "Loose Area": "Maples",
    "Mandalay": "Mandalay West",
    "Tyndall": "Tyndall Park",
    "Wolesley": "Wolseley",
    "Wolsey": "Wolseley",
    "Meadow": "Meadows",
}

KNOWN_NEIGHBORHOODS = [
    "Agassiz", "Airport", "Alpine Place", "Amber Trails", "Archwood", "Armstrong Point", "Assiniboia Downs", "Assiniboine Park",
    "Beaumont", "Betsworth", "Birchwood", "Booth", "Bridgwater Centre", "Bridgwater Forest", "Bridgwater Lakes", "Bridgwater Trails",
    "Broadway-Assiniboine", "Brockville", "Brooklands", "Bruce Park", "Buchanan", "Buffalo", "Burrows Central", "Burrows-Keewatin",
    "Canterbury Park", "Centennial", "Central Park", "Central River Heights", "Central St. Boniface", "Chalmers", "Chevrier", "China Town",
    "Civic Centre", "Cloutier Drive", "Colony", "Crescent Park", "Crescentwood", "Crestview",
    "Dakota Crossing", "Daniel McIntyre", "Deer Lodge", "Dufferin Industrial", "Dufferin", "Dufresne", "Dugald",
    "Eaglemere", "Earl Grey", "East Elmwood", "Ebby-Wentworth", "Edgeland", "Elm Park", "Elmhurst", "Eric Coy", "Exchange District",
    "Fairfield Park", "Fort Richmond", "Fraipont", "Garden City", "Glendale", "Glenelm", "Glenwood", "Grant Park", "Grassie", "Griffin",
    "Heritage Park", "Holden", "Inkster Gardens", "Inkster Industrial Park", "Inkster-Faraday", "Island Lakes",
    "J.B. Mitchell", "Jameswood", "Jefferson", "Kensington", "Kern Park", "Kil-cona Park", "Kildare-Redonda", "Kildonan Crossing",
    "Kildonan Drive", "Kildonan Park", "King Edward", "Kingston Crescent", "Kirkfield", "La Barriere", "Lavalee", "Legislature",
    "Leila North", "Leila-McPhillips Triangle", "Linden Ridge", "Linden Woods", "Logan-C.P.R", "Lord Roberts", "Lord Selkirk Park",
    "Luxton", "Maginot", "Mandalay West", "Maple Grove Park", "Margaret Park", "Marlton", "Mathers", "Maybank", "Mcleod Industrial",
    "McMillan", "Meadowood", "Meadows", "Melrose", "Minnetonka", "Minto", "Mission Gardens", "Mission Industrial", "Montcalm",
    "Munroe East", "Munroe West", "Murray Industrial Park", "Mynarski", "Niakwa Park", "Niakwa Place", "Norberry", "Normand Park",
    "North Inkster Industrial", "North Point Douglas", "North River Heights", "North St. Boniface", "North Transcona Yards",
    "Norwood East", "Norwood West", "Oak Point Highway", "Old Tuxedo", "Omand's Creek Industrial", "Pacific Industrial", "Parc La Salle",
    "Parker", "Peguis", "Pembina Strip", "Perrault", "Point Road", "Polo Park", "Portage & Main", "Portage-Ellice", "Prairie Pointe",
    "Pulberry", "Radisson", "Regent", "Richmond Lakes", "Richmond West", "Ridgedale", "Ridgewood South", "River East", "River Park South",
    "River West Park", "Riverbend", "Rivergrove", "River-Osborne", "Riverview", "Robertson", "Roblin Park", "Rockwood", "Roslyn",
    "Rosser-Old Kildonan", "Rossmere-A", "Rossmere-B", "Royalwood", "Sage Creek", "Sargent Park", "Saskatchewan North", "Seven Oaks",
    "Shaughnessy Park", "Silver Heights", "Sir John Franklin", "South Point Douglas", "South Pointe", "South Portage",
    "South River Heights", "South Tuxedo", "Southboine", "Southdale", "Southland Park", "Spence", "Springfield North", "Springfield South",
    "St. Boniface Industrial Park", "St. George", "St. James Industrial", "St. John's Park", "St. John's", "St. Matthews", "St. Norbert",
    "St. Vital Centre", "St. Vital Perimeter South", "Stock Yards", "Sturgeon Creek", "Symington Yards", "Talbot-Grey",
    "Templeton-Sinclair", "The Forks", "The Maples", "The Mint", "Tissot", "Transcona North", "Transcona South", "Transcona Yards",
    "Trappistes", "Turnbull Drive", "Tuxedo Industrial", "Tuxedo", "Tyndall Park", "Tyne-Tees", "University", "Valhalla", "Valley Gardens",
    "Varennes", "Varsity View", "Vialoux", "Victoria Crescent", "Victoria West", "Vista", "Waverley Heights", "Waverley West B",
    "Wellington Crescent", "West Alexander", "West Broadway", "West Fort Garry Industrial", "West Kildonan Industrial",
    "West Perimeter South", "West Wolseley", "Westdale", "Weston Shops", "Weston", "Westwood", "Whyte Ridge", "Wildwood",
    "Wilkes South", "William Whyte", "Windsor Park", "Wolseley", "Woodhaven", "Worthington"
]

STREET_TO_NEIGHBORHOOD = {
    "Arnold Ave": "Riverview", "Walker Ave": "Lord Roberts", "Ebby Ave": "Crescentwood", "Gertrude Ave": "Crescentwood",
    "Beresford Ave": "Lord Roberts", "Balfour Ave": "Riverview", "Clare Ave": "Riverview", "Rathgar Ave": "Lord Roberts",
    "Ashland Ave": "Riverview", "Fleet Ave": "Crescentwood", "Lorette Ave": "Crescentwood", "Lilac Street": "Crescentwood",
    "Mulvey Ave": "Crescentwood", "Dudley Ave": "Crescentwood", "Corydon Ave": "Crescentwood", "McMillan Ave": "Crescentwood",
    "Wavell Ave": "Riverview", "Morley Ave": "Riverview", "Churchill Drive": "Riverview", "Baltimore Road": "Riverview",
    "Norquay Street": "Crescentwood", "Hector Ave": "Crescentwood", "Hethrington Ave": "Lord Roberts",
}

# ---------- FREE OSM Geocoder Fallback ----------
GEOCODER_CACHE_FILE = "neighborhood_cache.json"

try:
    with open(GEOCODER_CACHE_FILE, "r") as f:
        neighborhood_cache = json.load(f)
except:
    neighborhood_cache = {}

def get_neighborhood_from_osm(address, city="Winnipeg", province="Manitoba", country="Canada"):
    key = f"{address}, {city}, {province}, {country}"
    if key in neighborhood_cache:
        return neighborhood_cache[key]
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": key, "format": "json", "addressdetails": 1, "limit": 1}
        headers = {"User-Agent": "SamVisionAI-Free-Geocoder"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        if data:
            address_details = data[0].get("address", {})
            neighborhood = (address_details.get("neighbourhood") or address_details.get("suburb") or
                            address_details.get("city_district") or address_details.get("city"))
            if neighborhood:
                neighborhood_cache[key] = neighborhood
                with open(GEOCODER_CACHE_FILE, "w") as f:
                    json.dump(neighborhood_cache, f)
                time.sleep(1)
                return neighborhood
    except Exception as e:
        print(f"🌐 Geocode failed for {key}: {e}")
    return None

# ---------- UTILITY FUNCTIONS ----------
def extract_text_from_path(path):
    return extract_text(path).split("Page ") if path.lower().endswith(".pdf") else open(path, encoding="utf-8").read().split("Page ")

def parse_currency(text):
    try:
        return float(text.replace("$", "").replace(",", "").strip())
    except:
        return None

def clean_address_field(addr):
    addr = re.sub(r"\s+", " ", addr.replace("\n", " ")).strip()
    return None if "Winnipeg Regional Real Estate Board" in addr else addr

def parse_bedrooms(text):
    match = re.search(r"BDA:\s*(\d+)\s*TBD:\s*(\d+)", text)
    if match:
        return int(match.group(2)) if int(match.group(2)) else int(match.group(1))
    alt_match = re.search(r"(\d+)\s*(?:bed|bdrm|bdrms)", text, re.IGNORECASE)
    return int(alt_match.group(1)) if alt_match else None

def parse_bathrooms(text):
    match = re.search(r"FB:\s*(\d+)\s*HB:\s*(\d+)", text)
    if match:
        return int(match.group(1)) + 0.5 * int(match.group(2))
    alt_match = re.search(r"(\d+(\.\d+)?)\s*(?:bath|baths)", text, re.IGNORECASE)
    return float(alt_match.group(1)) if alt_match else None

def normalize_garage_type(text):
    if not text:
        return "None"
    keywords = {"attached": "Attached", "detached": "Detached", "double": "Double", "single": "Single",
                "pad": "Pad", "rear": "Rear Drive", "plug": "Plug-In", "carport": "Carport"}
    found = [v for k, v in keywords.items() if k in text.lower()]
    return ", ".join(sorted(set(found))) if found else "None"

def parse_year_built_and_age(text):
    current_year = datetime.now().year
    match = re.search(r"(?:OL\s*)?/\s*([12][09]\d{2})", text)
    if match:
        yb = int(match.group(1))
        return yb, yb
    match_alt = re.search(r"(?:Year\s*)?Built[:\s]*([12][09]\d{2})", text, re.IGNORECASE)
    if match_alt:
        yb = int(match_alt.group(1))
        return yb, yb
    age_match = re.search(r"Age[:\s]*([0-9]{1,3})", text, re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))
        yb = current_year - age if age < 150 else None
        return yb, yb
    return None, None

import re

def parse_basement_type(text: str) -> str:
    """
    Robustly parse basement type from messy OCR / text blocks,
    preventing false 'Full' assignments, ignoring construction leakage,
    and ensuring a known basement type is returned.
    """
    text = text.lower()

    # Clean known leakage words that are not basement types
    leakage_words = [
        "bedroom", "bathroom", "electrical", "insulation", "kitchen", "furnace",
        "wood frame", "frame", "siding", "roof", "garage"
    ]
    for word in leakage_words:
        text = text.replace(word, "")

    # Priority matching for most accurate results
    if "crawl" in text:
        return "Crawl Space"
    if "3/4" in text or "three quarter" in text:
        return "Three Quarter"
    if "partial" in text:
        return "Partial"
    if "half" in text:
        return "Half"
    if "walkout" in text:
        return "Walkout"
    if "slab" in text:
        return "Slab"
    if "unfinished" in text:
        return "Unfinished"
    if "finished" in text:
        return "Finished"
    if "none" in text:
        return "None"
    if "full" in text:
        return "Full"

    # Fallback to "None" if no match
    return "None"

import re

# def parse_dom_days(page: str, idx: int) -> int:
#     """
#     Robust DOM extraction:
#     - Tolerates OCR splits and noisy lines
#     - Anchors on 'DOM' close to numbers
#     - Ignores outlier values (>180)
#     - Fallbacks gracefully
#     """
#     # Typical DOM range
#     MIN_DOM, MAX_DOM = 1, 180

#     # Step 1: Extract lines with DOM near a number
#     dom_candidates = []
#     for line in page.splitlines():
#         line_clean = line.strip()
#         if "DOM" in line_clean.upper():
#             numbers = re.findall(r"\d{1,3}", line_clean)
#             for num in numbers:
#                 n = int(num)
#                 if MIN_DOM <= n <= MAX_DOM:
#                     dom_candidates.append(n)

#     # Step 2: Use idx mapping if possible
#     if idx < len(dom_candidates):
#         return dom_candidates[idx]

#     # Step 3: Global fallback if structured parsing fails
#     # (Pick the most frequent value as fallback to reduce random noise)
#     if dom_candidates:
#         from collections import Counter
#         most_common = Counter(dom_candidates).most_common(1)[0][0]
#         return most_common

#     # Step 4: Final fallback
#     return 14


def parse_dom_days(page: str, idx: int, start: int) -> int:
    """
    Robust DOM extraction:
    - Anchors on 'DOM' or 'Days on Market'
    - Avoids misattribution to small numbers in OCR noise
    - Falls back safely
    """
    MIN_DOM, MAX_DOM = 1, 180
    block = page[max(0, start - 300): start + 1200]
    lines = block.splitlines()
    dom_candidates = []

    for line in lines:
        if re.search(r"\b(dom|days on market)\b", line, re.IGNORECASE):
            nums = re.findall(r"\b\d{1,3}\b", line)
            for n in nums:
                n_int = int(n)
                if MIN_DOM <= n_int <= MAX_DOM:
                    dom_candidates.append(n_int)

    if dom_candidates:
        from collections import Counter
        return Counter(dom_candidates).most_common(1)[0][0]

    # fallback: search globally in block with context
    context_matches = re.findall(r"(?:dom|days on market)[^\n]{0,40}?(\d{1,3})", block, re.IGNORECASE)
    for n in context_matches:
        n_int = int(n)
        if MIN_DOM <= n_int <= MAX_DOM:
            return n_int

    # fallback: median to reduce misreading
    fallback_nums = [int(x) for x in re.findall(r"\b\d{1,3}\b", block) if MIN_DOM <= int(x) <= MAX_DOM]
    if fallback_nums:
        fallback_nums.sort()
        return fallback_nums[len(fallback_nums)//2]

    return 14


def parse_neighborhood(page_text, block, address):
    combined = f"{address} {block} {page_text}".lower()
    combined = re.sub(r"[^\w\s]", " ", combined)
    combined = re.sub(r"\s+", " ", combined)
    for alias, canonical in NEIGHBORHOOD_ALIASES.items():
        if re.search(rf"\b{re.escape(alias.lower())}\b", combined):
            return canonical
    matched_hoods = [hood for hood in KNOWN_NEIGHBORHOODS if re.search(rf"\b{re.escape(hood.lower())}\b", combined)]
    if matched_hoods:
        return max(matched_hoods, key=len)
    if USE_FUZZY:
        words = combined.split()
        ngrams = [' '.join(words[i:i+ n]) for n in range(2, 6) for i in range(len(words) - n + 1)]
        best_score = 0
        best_match = None
        for ng in ngrams:
            result = process.extractOne(ng, KNOWN_NEIGHBORHOODS, scorer=fuzz.token_sort_ratio)
            if result:
                match, score, _ = result
                if score > best_score:
                    best_score = score
                    best_match = match
        if best_match and best_score > 75:
            return best_match
    # --- OSM fallback for Loose Area ---
    geo_neigh = get_neighborhood_from_osm(address)
    if geo_neigh:
        print(f"✅ OSM rescued Loose Area -> {geo_neigh} for {address}")
        return geo_neigh
    return "Loose Area"

# ---------- MAIN EXTRACTOR ----------
def extract_pdf_sales(path):
    pages = extract_text_from_path(path)
    today = datetime.today().date()
    season = ["Winter", "Spring", "Summer", "Fall"][(today.month % 12) // 3]
    records = []

    address_regex = re.compile(r"\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Ave|Avenue|Street|St|Drive|Dr|Road|Rd|Boulevard|Blvd|Lane|Ln|Bay|Crescent|Cres|Place|Pl|Parkway|Way|Trail|Court|Ct)", re.IGNORECASE)

    for page in pages:
        for idx, match in enumerate(address_regex.finditer(page)):
            start = match.start()
            block = page[start:start + 1500]
            address = clean_address_field(match.group())
            if not address:
                continue
            mls_match = re.search(r"\b20\d{6}\b", block)
            mls = mls_match.group(0) if mls_match else hashlib.md5(address.encode()).hexdigest()[:10]
            prices = [parse_currency(p) for p in re.findall(r"\$([\d,]+)", block)]
            prices = [p for p in prices if p and p > 30000]
            list_price = prices[0] if prices else None
            sold_price = prices[1] if len(prices) > 1 else None
            sell_ratio = round(sold_price / list_price, 2) if list_price and sold_price else None
            bedrooms = parse_bedrooms(block)
            bathrooms = parse_bathrooms(block)
            # basement_type = parse_basement_type(page, idx)
            # dom_days = parse_dom_days(page, idx)
            dom_days = parse_dom_days(page, idx, start)

            basement_type = parse_basement_type(block)

            listing_date = today

            sqft_match = re.search(r"([\d,]{3,5})\s*(?:sqft|sf|sq ft)", block, re.IGNORECASE)
            sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else None
            lot_match = re.search(r"Lot\s*Size[:\s]+(\d+)\s*[xX]\s*(\d+)", block, re.IGNORECASE)
            lot_size = round(int(lot_match.group(1)) * int(lot_match.group(2)) * 0.092903, 2) if lot_match else None
            garage_match = re.search(r"(attached|detached|double|single|pad|rear|plug|carport)[^\n]{0,50}", block, re.IGNORECASE)
            garage_type = normalize_garage_type(garage_match.group(0)) if garage_match else "None"
            style = next((s for s in HOUSE_STYLES if s.lower() in block.lower()), None)
            house_type = next((t for t in HOUSE_TYPES if t.lower() in block.lower()), "Single Family Detached")
            neighborhood = parse_neighborhood(page, block, address)
            year_built, _ = parse_year_built_and_age(block)

            records.append({
                "listing_date": listing_date,
                "season": season,
                "mls_number": mls,
                "neighborhood": neighborhood,
                "address": address,
                "list_price": list_price,
                "sold_price": sold_price,
                "sell_list_ratio": sell_ratio,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "dom_days": dom_days,
                "basement_type": basement_type,
                "garage_type": garage_type,
                "built_year": year_built,
                "house_type": house_type,
                "style": style,
                "sqft": sqft,
                "lot_size": lot_size,
            })


    return pd.DataFrame(records)
