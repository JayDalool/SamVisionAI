
# # file: utils/pdf_sales_parser.py

# import re
# import pandas as pd
# from datetime import datetime
# from typing import List
# from pdfminer.high_level import extract_text
# import os


# def extract_text_from_path(path: str) -> List[str]:
#     if path.lower().endswith(".pdf"):
#         text = extract_text(path)
#     else:
#         with open(path, "r", encoding="utf-8") as f:
#             text = f.read()
#     return text.split("Page ")


# def get_current_season():
#     return ["Winter", "Spring", "Summer", "Fall"][(datetime.today().month % 12) // 3]


# def parse_currency(value: str) -> float:
#     try:
#         return float(value.replace("$", "").replace(",", ""))
#     except:
#         return 0.0


# def parse_bedrooms(text: str) -> int:
#     match = re.search(r"BDA:\s*(\d+)", text)
#     return int(match.group(1)) if match else 3


# def parse_bathrooms(text: str) -> float:
#     fb = re.search(r"FB:\s*(\d+)", text)
#     hb = re.search(r"HB:\s*(\d+)", text)
#     return float(fb.group(1)) + 0.5 * float(hb.group(1)) if fb and hb else 2.0


# def normalize_garage_type(raw: str) -> str:
#     raw = raw.lower().strip() if raw else ""
#     if not raw or raw == 'none':
#         return "none"
#     tags = []
#     if "pad" in raw:
#         tags.append("Parking pad")
#     if "plug" in raw:
#         tags.append("plug-in")
#     if "rear" in raw:
#         tags.append("rear drive")
#     if "unpaved" in raw:
#         tags.append("unpaved")
#     if "attached" in raw:
#         tags.append("attached")
#     if "detached" in raw:
#         tags.append("detached")
#     if "single" in raw:
#         tags.append("single")
#     if "double" in raw:
#         tags.append("double")
#     if "carport" in raw:
#         tags.append("carport")
#     return ", ".join(sorted(set(tags))) if tags else raw


# def clean_address_field(addr: str) -> str:
#     addr = re.sub(r"\n", " ", addr)
#     addr = re.sub(r"Winnipeg Regional.*?Levies", "", addr, flags=re.IGNORECASE)
#     addr = re.sub(r"\d+\s+Sold", "", addr)
#     addr = re.sub(r"\s+", " ", addr).strip()
#     return addr


# def extract_pdf_sales(path: str) -> pd.DataFrame:
#     pages = extract_text_from_path(path)
#     records = []

#     known_neighborhoods = [
#         "Riverview", "Lord Roberts", "Fort Rouge", "Osborne Village", "Crescentwood",
#         "Charleswood", "Richmond West", "Fairfield Park", "Deer Pointe", "Headingley South",
#         "St Boniface", "Norwood", "Norwood Flats", "East Fort Garry", "Wildwood",
#         "East Kildonan", "St Vital", "St Norbert", "St James", "Tuxedo",
#         "River Heights", "Linden Woods", "Garden City", "Transcona", "West End"
#     ]

#     for page_text in pages:
#         entries = re.split(r"(?=Sold\d{8})", page_text)
#         for listing in entries:
#             if not listing.strip():
#                 continue

#             address_match = re.search(r"(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", listing)
#             mls_match = re.search(r"MLS[\u00aeR#\s:]*?(\d{8})", listing)
#             dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
#             sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
#             lot_match = re.search(r"(\d+\.\d+)\s*M2", listing)
#             price_matches = re.findall(r"\$(\d{1,3}(?:,\d{3})*)", listing)

#             raw_addr = address_match.group(1).strip() if address_match else f"Unknown Address"
#             address = clean_address_field(raw_addr)
#             address_lower = address.lower()
#             listing_lower = listing.lower()

#             neighborhood = "Loose Area"
#             for hood in known_neighborhoods:
#                 hood_lower = hood.lower()
#                 if hood_lower in listing_lower or hood_lower in address_lower:
#                     neighborhood = hood
#                     break

#             if neighborhood == "Loose Area":
#                 suffix_map = {
#                     "beresford": "Lord Roberts",
#                     "arnold": "Riverview",
#                     "wardlaw": "Osborne Village",
#                     "mcmillan": "Crescentwood",
#                     "fleet": "Osborne Village",
#                     "walker": "Lord Roberts",
#                     "ashland": "Riverview",
#                     "balfour": "Riverview",
#                     "montgomery": "Riverview",
#                     "morley": "Lord Roberts",
#                     "clark": "Riverview",
#                     "baltimore": "Riverview",
#                     "churchill": "Fort Rouge",
#                     "dudley": "Crescentwood",
#                     "mulvey": "Crescentwood",
#                     "scotland": "Crescentwood",
#                     "warsaw": "Crescentwood",
#                     "corydon": "Crescentwood",
#                     "hector": "Crescentwood",
#                     "gauvin": "St Boniface",
#                     "archibald": "St Boniface",
#                     "deniset": "St Boniface",
#                     "lariviere": "St Boniface",
#                     "lyndale": "Norwood Flats",
#                     "eugenie": "Norwood Flats",
#                     "redview": "St Vital",
#                     "kitson": "St Boniface",
#                     "dollard": "St Boniface",
#                     "langevin": "St Boniface"
#                 }
#                 for suffix, hood in suffix_map.items():
#                     if suffix in address_lower:
#                         neighborhood = hood
#                         break

#             mls_number = mls_match.group(1) if mls_match else "unknown"
#             dom = int(dom_match.group(1)) if dom_match else 0
#             sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200

#             lot_size = 0.0
#             if lot_match:
#                 lot_val = lot_match.group(1)
#                 try:
#                     lot_size = float(lot_val)
#                 except:
#                     lot_size = 0.0

#             list_price = parse_currency(price_matches[-4]) if len(price_matches) >= 4 else 0.0
#             sold_price = parse_currency(price_matches[-2]) if len(price_matches) >= 4 else 0.0

#             garage_match = re.search(r"Parking[\s\S]{0,100}?((?:Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", listing, re.IGNORECASE)
#             garage_raw = garage_match.group(1) if garage_match else ""
#             garage_type = normalize_garage_type(garage_raw)

#             house_type_match = re.search(r"Property Type\s*:?.*?(Single Family Detached|Townhouse|Bungalow|Duplex|Condo|Mobile|Other)", listing, re.IGNORECASE)
#             house_type = house_type_match.group(1).strip() if house_type_match else "Single Family Detached"
#             style_match = re.search(r"Style\s*([^\n]*)", listing, re.IGNORECASE)
#             style = style_match.group(1).strip() if style_match else ""

#             type_match = re.search(r"Type\s*([^\n]*)", listing, re.IGNORECASE)
#             type_ = type_match.group(1).strip() if type_match else ""


#             bedrooms = parse_bedrooms(listing)
#             bathrooms = parse_bathrooms(listing)

#             records.append({
#                 "listing_date": datetime.today().date(),
#                 "season": get_current_season(),
#                 "mls_number": mls_number,
#                 "neighborhood": neighborhood,
#                 "address": address,
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
#                 "dom": dom,
#                 "bedrooms": bedrooms,
#                 "bathrooms": bathrooms,
#                 "garage_type": garage_type,
#                 "house_type": house_type,
#                 "style": style,
#                 "type": type_,
#                 "sqft": sqft,
#                 "lot_size": lot_size,
#             })

#     return pd.DataFrame(records)

# file: utils/pdf_sales_parser.py

# import re
# import pandas as pd
# from datetime import datetime
# from typing import List
# from pdfminer.high_level import extract_text
# import os


# HOUSE_STYLES = [
#     "Bungalow", "One and a Half", "Two Storey", "Bi-Level", "Split-Level",
#     "Cab-Over", "Multi-Level", "Three Storey", "Back Split", "Side Split"
# ]

# HOUSE_TYPES = [
#     "Single Family Detached", "Single Family Attached", "Condo", "Townhouse",
#     "Duplex", "Triplex", "Fourplex", "Mobile Home", "Other"
# ]


# def extract_text_from_path(path: str) -> List[str]:
#     if path.lower().endswith(".pdf"):
#         text = extract_text(path)
#     else:
#         with open(path, "r", encoding="utf-8") as f:
#             text = f.read()
#     return text.split("Page ")


# def get_current_season():
#     return ["Winter", "Spring", "Summer", "Fall"][(datetime.today().month % 12) // 3]


# def parse_currency(value: str) -> float:
#     try:
#         return float(value.replace("$", "").replace(",", ""))
#     except:
#         return 0.0


# def parse_bedrooms(text: str) -> int:
#     match = re.search(r"BDA:\s*(\d+)", text)
#     return int(match.group(1)) if match else 3


# def parse_bathrooms(text: str) -> float:
#     fb = re.search(r"FB:\s*(\d+)", text)
#     hb = re.search(r"HB:\s*(\d+)", text)
#     return float(fb.group(1)) + 0.5 * float(hb.group(1)) if fb and hb else 2.0


# def normalize_garage_type(raw: str) -> str:
#     raw = raw.lower().strip() if raw else ""
#     if not raw or raw == 'none':
#         return "none"
#     tags = []
#     if "pad" in raw:
#         tags.append("Parking pad")
#     if "plug" in raw:
#         tags.append("plug-in")
#     if "rear" in raw:
#         tags.append("rear drive")
#     if "unpaved" in raw:
#         tags.append("unpaved")
#     if "attached" in raw:
#         tags.append("attached")
#     if "detached" in raw:
#         tags.append("detached")
#     if "single" in raw:
#         tags.append("single")
#     if "double" in raw:
#         tags.append("double")
#     if "carport" in raw:
#         tags.append("carport")
#     return ", ".join(sorted(set(tags))) if tags else raw


# def clean_address_field(addr: str) -> str:
#     addr = re.sub(r"\n", " ", addr)
#     addr = re.sub(r"Winnipeg Regional.*?Levies", "", addr, flags=re.IGNORECASE)
#     addr = re.sub(r"\d+\s+Sold", "", addr)
#     addr = re.sub(r"\s+", " ", addr).strip()
#     return addr


# def match_known_terms(text: str, known_terms: List[str]) -> str:
#     found = []
#     for term in known_terms:
#         if term.lower() in text.lower():
#             found.append(term)
#     return ", ".join(found)


# def extract_pdf_sales(path: str) -> pd.DataFrame:
#     pages = extract_text_from_path(path)
#     records = []

#     known_neighborhoods = [
#         "Riverview", "Lord Roberts", "Fort Rouge", "Osborne Village", "Crescentwood",
#         "Charleswood", "Richmond West", "Fairfield Park", "Deer Pointe", "Headingley South",
#         "St Boniface", "Norwood", "Norwood Flats", "East Fort Garry", "Wildwood",
#         "East Kildonan", "St Vital", "St Norbert", "St James", "Tuxedo",
#         "River Heights", "Linden Woods", "Garden City", "Transcona", "West End"
#     ]

#     for page_text in pages:
#         entries = re.split(r"(?=Sold\d{8})", page_text)
#         for listing in entries:
#             split_parts = re.split(r"(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", listing)
#             for i in range(1, len(split_parts), 2):
#                 segment = split_parts[i] + split_parts[i+1] if i + 1 < len(split_parts) else split_parts[i]
#                 if not segment.strip():
#                     continue

#                 address_match = re.search(r"(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", segment)
#                 mls_match = re.search(r"MLS[\u00aeR#\s:]*?(\d{8})", segment)
#                 dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", segment, re.IGNORECASE)
#                 sqft_match = re.search(r"(\d{3,5})\s+SF", segment)
#                 lot_match = re.search(r"(\d+\.\d+)\s*M2", segment)
#                 price_matches = re.findall(r"\$(\d{1,3}(?:,\d{3})*)", segment)

#                 raw_addr = address_match.group(1).strip() if address_match else f"Unknown Address"
#                 address = clean_address_field(raw_addr)
#                 address_lower = address.lower()
#                 segment_lower = segment.lower()

#                 neighborhood = "Loose Area"
#                 for hood in known_neighborhoods:
#                     hood_lower = hood.lower()
#                     if hood_lower in segment_lower or hood_lower in address_lower:
#                         neighborhood = hood
#                         break

#                 mls_number = mls_match.group(1) if mls_match else "unknown"
#                 dom = int(dom_match.group(1)) if dom_match else 0
#                 sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200

#                 lot_size = 0.0
#                 if lot_match:
#                     lot_val = lot_match.group(1)
#                     try:
#                         lot_size = float(lot_val)
#                     except:
#                         lot_size = 0.0

#                 list_price = parse_currency(price_matches[-4]) if len(price_matches) >= 4 else 0.0
#                 sold_price = parse_currency(price_matches[-2]) if len(price_matches) >= 4 else 0.0

#                 garage_match = re.search(r"Parking[\s\S]{0,100}?((?:Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", segment, re.IGNORECASE)
#                 garage_raw = garage_match.group(1) if garage_match else ""
#                 garage_type = normalize_garage_type(garage_raw)

#                 style = match_known_terms(segment, HOUSE_STYLES)
#                 type_ = match_known_terms(segment, HOUSE_TYPES)

#                 house_type_match = re.search(r"Property Type\s*:?.*?(Single Family Detached|Townhouse|Bungalow|Duplex|Condo|Mobile|Other)", segment, re.IGNORECASE)
#                 house_type = house_type_match.group(1).strip() if house_type_match else "Single Family Detached"

#                 bedrooms = parse_bedrooms(segment)
#                 bathrooms = parse_bathrooms(segment)

#                 records.append({
#                     "listing_date": datetime.today().date(),
#                     "season": get_current_season(),
#                     "mls_number": mls_number,
#                     "neighborhood": neighborhood,
#                     "address": address,
#                     "list_price": list_price,
#                     "sold_price": sold_price,
#                     "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
#                     "dom": dom,
#                     "bedrooms": bedrooms,
#                     "bathrooms": bathrooms,
#                     "garage_type": garage_type,
#                     "house_type": house_type,
#                     "style": style,
#                     "type": type_,
#                     "sqft": sqft,
#                     "lot_size": lot_size,
#                 })

#     return pd.DataFrame(records)


# # file: utils/pdf_sales_parser.py

# import re
# import pandas as pd
# from datetime import datetime
# from typing import List
# from pdfminer.high_level import extract_text
# import os


# def extract_text_from_path(path: str) -> List[str]:
#     if path.lower().endswith(".pdf"):
#         text = extract_text(path)
#     else:
#         with open(path, "r", encoding="utf-8") as f:
#             text = f.read()
#     return text.split("Page ")


# def get_current_season():
#     return ["Winter", "Spring", "Summer", "Fall"][(datetime.today().month % 12) // 3]


# def parse_currency(value: str) -> float:
#     try:
#         return float(value.replace("$", "").replace(",", ""))
#     except:
#         return 0.0


# def parse_bedrooms(text: str) -> int:
#     match = re.search(r"BDA:\s*(\d+)", text)
#     return int(match.group(1)) if match else 3


# def parse_bathrooms(text: str) -> float:
#     fb = re.search(r"FB:\s*(\d+)", text)
#     hb = re.search(r"HB:\s*(\d+)", text)
#     return float(fb.group(1)) + 0.5 * float(hb.group(1)) if fb and hb else 2.0


# def normalize_garage_type(raw: str) -> str:
#     raw = raw.lower().strip() if raw else ""
#     if not raw or raw == 'none':
#         return "none"
#     tags = []
#     if "pad" in raw:
#         tags.append("Parking pad")
#     if "plug" in raw:
#         tags.append("plug-in")
#     if "rear" in raw:
#         tags.append("rear drive")
#     if "unpaved" in raw:
#         tags.append("unpaved")
#     if "attached" in raw:
#         tags.append("attached")
#     if "detached" in raw:
#         tags.append("detached")
#     if "single" in raw:
#         tags.append("single")
#     if "double" in raw:
#         tags.append("double")
#     if "carport" in raw:
#         tags.append("carport")
#     return ", ".join(sorted(set(tags))) if tags else raw


# def clean_address_field(addr: str) -> str:
#     addr = re.sub(r"\n", " ", addr)
#     addr = re.sub(r"Winnipeg Regional.*?Levies", "", addr, flags=re.IGNORECASE)
#     addr = re.sub(r"\d+\s+Sold", "", addr)
#     addr = re.sub(r"\s+", " ", addr).strip()
#     return addr


# def extract_pdf_sales(path: str) -> pd.DataFrame:
#     pages = extract_text_from_path(path)
#     records = []

#     known_neighborhoods = [
#         "Riverview", "Lord Roberts", "Fort Rouge", "Osborne Village", "Crescentwood",
#         "Charleswood", "Richmond West", "Fairfield Park", "Deer Pointe", "Headingley South",
#         "St Boniface", "Norwood", "Norwood Flats", "East Fort Garry", "Wildwood",
#         "East Kildonan", "St Vital", "St Norbert", "St James", "Tuxedo",
#         "River Heights", "Linden Woods", "Garden City", "Transcona", "West End"
#     ]

#     for page_text in pages:
#         entries = re.split(r"(?=Sold\d{8})", page_text)
#         for listing in entries:
#             if not listing.strip():
#                 continue

#             address_match = re.search(r"(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", listing)
#             mls_match = re.search(r"MLS[\u00aeR#\s:]*?(\d{8})", listing)
#             dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
#             sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
#             lot_match = re.search(r"(\d+\.\d+)\s*M2", listing)
#             price_matches = re.findall(r"\$(\d{1,3}(?:,\d{3})*)", listing)

#             raw_addr = address_match.group(1).strip() if address_match else f"Unknown Address"
#             address = clean_address_field(raw_addr)
#             address_lower = address.lower()
#             listing_lower = listing.lower()

#             neighborhood = "Loose Area"
#             for hood in known_neighborhoods:
#                 hood_lower = hood.lower()
#                 if hood_lower in listing_lower or hood_lower in address_lower:
#                     neighborhood = hood
#                     break

#             if neighborhood == "Loose Area":
#                 suffix_map = {
#                     "beresford": "Lord Roberts",
#                     "arnold": "Riverview",
#                     "wardlaw": "Osborne Village",
#                     "mcmillan": "Crescentwood",
#                     "fleet": "Osborne Village",
#                     "walker": "Lord Roberts",
#                     "ashland": "Riverview",
#                     "balfour": "Riverview",
#                     "montgomery": "Riverview",
#                     "morley": "Lord Roberts",
#                     "clark": "Riverview",
#                     "baltimore": "Riverview",
#                     "churchill": "Fort Rouge",
#                     "dudley": "Crescentwood",
#                     "mulvey": "Crescentwood",
#                     "scotland": "Crescentwood",
#                     "warsaw": "Crescentwood",
#                     "corydon": "Crescentwood",
#                     "hector": "Crescentwood",
#                     "gauvin": "St Boniface",
#                     "archibald": "St Boniface",
#                     "deniset": "St Boniface",
#                     "lariviere": "St Boniface",
#                     "lyndale": "Norwood Flats",
#                     "eugenie": "Norwood Flats",
#                     "redview": "St Vital",
#                     "kitson": "St Boniface",
#                     "dollard": "St Boniface",
#                     "langevin": "St Boniface"
#                 }
#                 for suffix, hood in suffix_map.items():
#                     if suffix in address_lower:
#                         neighborhood = hood
#                         break

#             mls_number = mls_match.group(1) if mls_match else "unknown"
#             dom = int(dom_match.group(1)) if dom_match else 0
#             sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200

#             lot_size = 0.0
#             if lot_match:
#                 lot_val = lot_match.group(1)
#                 try:
#                     lot_size = float(lot_val)
#                 except:
#                     lot_size = 0.0

#             list_price = parse_currency(price_matches[-4]) if len(price_matches) >= 4 else 0.0
#             sold_price = parse_currency(price_matches[-2]) if len(price_matches) >= 4 else 0.0

#             garage_match = re.search(r"Parking[\s\S]{0,100}?((?:Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", listing, re.IGNORECASE)
#             garage_raw = garage_match.group(1) if garage_match else ""
#             garage_type = normalize_garage_type(garage_raw)

#             house_type_match = re.search(r"Property Type\s*:?.*?(Single Family Detached|Townhouse|Bungalow|Duplex|Condo|Mobile|Other)", listing, re.IGNORECASE)
#             house_type = house_type_match.group(1).strip() if house_type_match else "Single Family Detached"
#             style_match = re.search(r"Style\s*([^\n]*)", listing, re.IGNORECASE)
#             style = style_match.group(1).strip() if style_match else ""

#             type_match = re.search(r"Type\s*([^\n]*)", listing, re.IGNORECASE)
#             type_ = type_match.group(1).strip() if type_match else ""


#             bedrooms = parse_bedrooms(listing)
#             bathrooms = parse_bathrooms(listing)

#             records.append({
#                 "listing_date": datetime.today().date(),
#                 "season": get_current_season(),
#                 "mls_number": mls_number,
#                 "neighborhood": neighborhood,
#                 "address": address,
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
#                 "dom": dom,
#                 "bedrooms": bedrooms,
#                 "bathrooms": bathrooms,
#                 "garage_type": garage_type,
#                 "house_type": house_type,
#                 "style": style,
#                 "type": type_,
#                 "sqft": sqft,
#                 "lot_size": lot_size,
#             })

#     return pd.DataFrame(records)

# file: utils/pdf_sales_parser.py
# # file: utils/pdf_sales_parser.py

# import re
# import pandas as pd
# from datetime import datetime
# from typing import List
# from pdfminer.high_level import extract_text
# import os
# import hashlib


# HOUSE_STYLES = [
#     "Bungalow", "One and a Half", "Two Storey", "Bi-Level", "Split-Level",
#     "Cab-Over", "Multi-Level", "Three Storey", "Back Split", "Side Split"
# ]

# HOUSE_TYPES = [
#     "Single Family Detached", "Single Family Attached", "Condo", "Townhouse",
#     "Duplex", "Triplex", "Fourplex", "Mobile Home", "Other"
# ]


# class SalesCleaner:
#     @staticmethod
#     def clean(df: pd.DataFrame) -> pd.DataFrame:
#         df = df[df['neighborhood'] != "Loose Area"]
#         df = df.drop_duplicates(subset=['address'], keep='last')
#         return df


# def extract_text_from_path(path: str) -> List[str]:
#     if path.lower().endswith(".pdf"):
#         text = extract_text(path)
#     else:
#         with open(path, "r", encoding="utf-8") as f:
#             text = f.read()
#     return text.split("Page ")


# def get_current_season():
#     return ["Winter", "Spring", "Summer", "Fall"][(datetime.today().month % 12) // 3]


# def parse_currency(value: str) -> float:
#     try:
#         return float(value.replace("$", "").replace(",", ""))
#     except:
#         return 0.0


# def parse_bedrooms(text: str) -> int:
#     match = re.search(r"BDA:\s*(\d+)", text)
#     return int(match.group(1)) if match else 3


# def parse_bathrooms(text: str) -> float:
#     fb = re.search(r"FB:\s*(\d+)", text)
#     hb = re.search(r"HB:\s*(\d+)", text)
#     return float(fb.group(1)) + 0.5 * float(hb.group(1)) if fb and hb else 2.0


# def normalize_garage_type(raw: str) -> str:
#     raw = raw.lower().strip() if raw else ""
#     if not raw or raw == 'none':
#         return "none"
#     tags = []
#     if "pad" in raw:
#         tags.append("Parking pad")
#     if "plug" in raw:
#         tags.append("plug-in")
#     if "rear" in raw:
#         tags.append("rear drive")
#     if "unpaved" in raw:
#         tags.append("unpaved")
#     if "attached" in raw:
#         tags.append("attached")
#     if "detached" in raw:
#         tags.append("detached")
#     if "single" in raw:
#         tags.append("single")
#     if "double" in raw:
#         tags.append("double")
#     if "carport" in raw:
#         tags.append("carport")
#     return ", ".join(sorted(set(tags))) if tags else raw


# def clean_address_field(addr: str) -> str:
#     addr = re.sub(r"\n", " ", addr)
#     addr = re.sub(r"Winnipeg Regional.*?Levies", "", addr, flags=re.IGNORECASE)
#     addr = re.sub(r"\d+\s+Sold", "", addr)
#     addr = re.sub(r"\s+", " ", addr).strip()
#     return addr


# def match_known_terms(text: str, known_terms: List[str]) -> str:
#     found = []
#     for term in known_terms:
#         if term.lower() in text.lower():
#             found.append(term)
#     return ", ".join(found)


# def extract_pdf_sales(path: str) -> pd.DataFrame:
#     pages = extract_text_from_path(path)
#     records = []

#     known_neighborhoods = [
#         "Riverview", "Lord Roberts", "Fort Rouge", "Osborne Village", "Crescentwood",
#         "Charleswood", "Richmond West", "Fairfield Park", "Deer Pointe", "Headingley South",
#         "St Boniface", "Norwood", "Norwood Flats", "East Fort Garry", "Wildwood",
#         "East Kildonan", "St Vital", "St Norbert", "St James", "Tuxedo",
#         "River Heights", "Linden Woods", "Garden City", "Transcona", "West End"
#     ]

#     for page_text in pages:
#         entries = re.findall(
#             r"(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place)[\s\S]*?)(?=\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place)|\Z)",
#             page_text
#         )
#         for listing in entries:
#             if not listing.strip():
#                 continue

#             address_match = re.search(r"(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", listing)
#             mls_match = re.search(r"MLS[\u00aeR#\s:]*?(\d{8})", listing)
#             dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
#             sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
#             lot_match = re.search(r"(\d+\.\d+)\s*M2", listing)
#             price_matches = re.findall(r"\$(\d{1,3}(?:,\d{3})*)", listing)

#             raw_addr = address_match.group(1).strip() if address_match else f"Unknown Address"
#             address = clean_address_field(raw_addr)
#             address_lower = address.lower()
#             listing_lower = listing.lower()

#             neighborhood = "Loose Area"
#             for hood in known_neighborhoods:
#                 hood_lower = hood.lower()
#                 if hood_lower in listing_lower or hood_lower in address_lower:
#                     neighborhood = hood
#                     break

#             if neighborhood == "Loose Area":
#                 suffix_map = {
#                     "beresford": "Lord Roberts",
#                     "arnold": "Riverview",
#                     "wardlaw": "Osborne Village",
#                     "mcmillan": "Crescentwood",
#                     "fleet": "Osborne Village",
#                     "walker": "Lord Roberts",
#                     "ashland": "Riverview",
#                     "balfour": "Riverview",
#                     "montgomery": "Riverview",
#                     "morley": "Lord Roberts",
#                     "clark": "Riverview",
#                     "baltimore": "Riverview",
#                     "churchill": "Fort Rouge",
#                     "dudley": "Crescentwood",
#                     "mulvey": "Crescentwood",
#                     "scotland": "Crescentwood",
#                     "warsaw": "Crescentwood",
#                     "corydon": "Crescentwood",
#                     "hector": "Crescentwood",
#                     "gauvin": "St Boniface",
#                     "archibald": "St Boniface",
#                     "deniset": "St Boniface",
#                     "lariviere": "St Boniface",
#                     "lyndale": "Norwood Flats",
#                     "eugenie": "Norwood Flats",
#                     "redview": "St Vital",
#                     "kitson": "St Boniface",
#                     "dollard": "St Boniface",
#                     "langevin": "St Boniface"
#                 }
#                 for suffix, hood in suffix_map.items():
#                     if suffix in address_lower:
#                         neighborhood = hood
#                         break

#             mls_number = mls_match.group(1) if mls_match else "unknown"
#             if mls_number == "unknown":
#                 mls_number = hashlib.md5(address.encode()).hexdigest()[:10]

#             dom = int(dom_match.group(1)) if dom_match else 0
#             sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200

#             lot_size = 0.0
#             if lot_match:
#                 lot_val = lot_match.group(1)
#                 try:
#                     lot_size = float(lot_val)
#                 except:
#                     lot_size = 0.0

#             list_price = parse_currency(price_matches[-4]) if len(price_matches) >= 4 else 0.0
#             sold_price = parse_currency(price_matches[-2]) if len(price_matches) >= 4 else 0.0

#             garage_match = re.search(r"Parking[\s\S]{0,100}?((?:Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", listing, re.IGNORECASE)
#             garage_raw = garage_match.group(1) if garage_match else ""
#             garage_type = normalize_garage_type(garage_raw)

#             style = match_known_terms(listing, HOUSE_STYLES)
#             type_ = match_known_terms(listing, HOUSE_TYPES)
#             if "Other" in type_ and len(type_.split(",")) > 1:
#                 type_ = ", ".join([t for t in type_.split(",") if t.strip() != "Other"])

#             house_type_match = re.search(r"Property Type\s*:?.*?(Single Family Detached|Townhouse|Bungalow|Duplex|Condo|Mobile|Other)", listing, re.IGNORECASE)
#             house_type = house_type_match.group(1).strip() if house_type_match else "Single Family Detached"

#             bedrooms = parse_bedrooms(listing)
#             bathrooms = parse_bathrooms(listing)

#             records.append({
#                 "listing_date": datetime.today().date(),
#                 "season": get_current_season(),
#                 "mls_number": mls_number,
#                 "neighborhood": neighborhood,
#                 "address": address,
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
#                 "dom": dom,
#                 "bedrooms": bedrooms,
#                 "bathrooms": bathrooms,
#                 "garage_type": garage_type,
#                 "house_type": house_type,
#                 "style": style,
#                 "type": type_,
#                 "sqft": sqft,
#                 "lot_size": lot_size,
#             })

#     df = pd.DataFrame(records)
#     return SalesCleaner.clean(df)
# file: utils/pdf_sales_parser.py

import re
import pandas as pd
from datetime import datetime
from typing import List
from pdfminer.high_level import extract_text
import os
import hashlib


HOUSE_STYLES = [
    "Bungalow", "One and a Half", "Two Storey", "Bi-Level", "Split-Level",
    "Cab-Over", "Multi-Level", "Three Storey", "Back Split", "Side Split"
]

HOUSE_TYPES = [
    "Single Family Detached", "Single Family Attached", "Condo", "Townhouse",
    "Duplex", "Triplex", "Fourplex", "Mobile Home", "Other"
]


class SalesCleaner:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['neighborhood'] != "Loose Area"]
        df = df.drop_duplicates(subset=['address'], keep='last')
        return df


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


def match_known_terms(text: str, known_terms: List[str]) -> str:
    found = []
    for term in known_terms:
        if term.lower() in text.lower():
            found.append(term)
    return ", ".join(found)


def backfill_missing_fields_from_txt(df: pd.DataFrame, txt_paths: List[str]) -> pd.DataFrame:
    for idx, row in df.iterrows():
        if row['style'] and row['type'] and row['garage_type'] != 'none':
            continue
        for path in txt_paths:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                pattern = re.escape(row['address'])[:20]
                match = re.search(rf"({pattern}[\s\S]{{0,1000}})", content, re.IGNORECASE)
                if match:
                    snippet = match.group(1)
                    style = match_known_terms(snippet, HOUSE_STYLES)
                    type_ = match_known_terms(snippet, HOUSE_TYPES)
                    garage_match = re.search(r"Parking[\s\S]{0,100}?((?:Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", snippet, re.IGNORECASE)
                    garage_raw = garage_match.group(1) if garage_match else ""
                    df.at[idx, 'style'] = style or row['style']
                    df.at[idx, 'type'] = type_ or row['type']
                    df.at[idx, 'garage_type'] = normalize_garage_type(garage_raw) or row['garage_type']
    return df


def get_txt_siblings(pdf_path: str) -> List[str]:
    base = os.path.splitext(pdf_path)[0]
    directory = os.path.dirname(pdf_path)
    candidates = [f for f in os.listdir(directory) if f.endswith(".txt")]
    return [os.path.join(directory, f) for f in candidates if base in f]




# MAIN PARSER

# MAIN PARSER

# def extract_pdf_sales(path: str) -> pd.DataFrame:
#     pages = extract_text_from_path(path)
#     records = []

#     known_neighborhoods = [
#         "Riverview", "Lord Roberts", "Fort Rouge", "Osborne Village", "Crescentwood",
#         "Charleswood", "Richmond West", "Fairfield Park", "Deer Pointe", "Headingley South",
#         "St Boniface", "Norwood", "Norwood Flats", "East Fort Garry", "Wildwood",
#         "East Kildonan", "St Vital", "St Norbert", "St James", "Tuxedo",
#         "River Heights", "Linden Woods", "Garden City", "Transcona", "West End"
#     ]

#     suffix_map = {
#         "beresford": "Lord Roberts",
#         "arnold": "Riverview",
#         "wardlaw": "Osborne Village",
#         "mcmillan": "Crescentwood",
#         "fleet": "Osborne Village",
#         "walker": "Lord Roberts",
#         "ashland": "Riverview",
#         "balfour": "Riverview",
#         "montgomery": "Riverview",
#         "morley": "Lord Roberts",
#         "clark": "Riverview",
#         "baltimore": "Riverview",
#         "churchill": "Fort Rouge",
#         "dudley": "Crescentwood",
#         "mulvey": "Crescentwood",
#         "scotland": "Crescentwood",
#         "warsaw": "Crescentwood",
#         "corydon": "Crescentwood",
#         "hector": "Crescentwood",
#         "gauvin": "St Boniface",
#         "archibald": "St Boniface",
#         "deniset": "St Boniface",
#         "lariviere": "St Boniface",
#         "lyndale": "Norwood Flats",
#         "eugenie": "Norwood Flats",
#         "redview": "St Vital",
#         "kitson": "St Boniface",
#         "dollard": "St Boniface",
#         "langevin": "St Boniface"
#     }

#     address_regex = re.compile(r"\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place)")

#     for page_text in pages:
#         address_starts = [(m.start(), m.group()) for m in address_regex.finditer(page_text)]

#         for i, (start, _) in enumerate(address_starts):
#             end = address_starts[i + 1][0] if i + 1 < len(address_starts) else len(page_text)
#             listing = page_text[start:end].strip()

#             address_match = address_regex.search(listing)
#             mls_match = re.search(r"MLS[\u00aeR#\s:]*?(\d{8})", listing)
#             dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
#             sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
#             lot_match = re.search(r"(\d+\.\d+)\s*M2", listing)
#             price_matches = re.findall(r"\$(\d{1,3}(?:,\d{3})*)", listing)

#             raw_addr = address_match.group().strip() if address_match else f"Unknown Address"
#             address = clean_address_field(raw_addr)
#             address_lower = address.lower()
#             listing_lower = listing.lower()

#             neighborhood = "Loose Area"
#             for hood in known_neighborhoods:
#                 if hood.lower() in listing_lower or hood.lower() in address_lower:
#                     neighborhood = hood
#                     break

#             if neighborhood == "Loose Area":
#                 for suffix, hood in suffix_map.items():
#                     if suffix in address_lower:
#                         neighborhood = hood
#                         break

#             mls_number = mls_match.group(1) if mls_match else hashlib.md5(address.encode()).hexdigest()[:10]
#             dom = int(dom_match.group(1)) if dom_match else 0
#             sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200

#             lot_size = float(lot_match.group(1)) if lot_match else 0.0
#             list_price = parse_currency(price_matches[-4]) if len(price_matches) >= 4 else 0.0
#             sold_price = parse_currency(price_matches[-2]) if len(price_matches) >= 4 else 0.0

#             garage_match = re.search(r"Parking[\s\S]{0,100}?((?:Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", listing, re.IGNORECASE)
#             garage_raw = garage_match.group(1) if garage_match else ""
#             garage_type = normalize_garage_type(garage_raw)

#             style = match_known_terms(listing, HOUSE_STYLES)
#             type_ = match_known_terms(listing, HOUSE_TYPES)
#             if "Other" in type_ and len(type_.split(",")) > 1:
#                 type_ = ", ".join([t for t in type_.split(",") if t.strip() != "Other"])

#             house_type_match = re.search(r"Property Type\s*:?.*?(Single Family Detached|Townhouse|Bungalow|Duplex|Condo|Mobile|Other)", listing, re.IGNORECASE)
#             house_type = house_type_match.group(1).strip() if house_type_match else "Single Family Detached"

#             bedrooms = parse_bedrooms(listing)
#             bathrooms = parse_bathrooms(listing)

#             records.append({
#                 "listing_date": datetime.today().date(),
#                 "season": get_current_season(),
#                 "mls_number": mls_number,
#                 "neighborhood": neighborhood,
#                 "address": address,
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
#                 "dom": dom,
#                 "bedrooms": bedrooms,
#                 "bathrooms": bathrooms,
#                 "garage_type": garage_type,
#                 "house_type": house_type,
#                 "style": style,
#                 "type": type_,
#                 "sqft": sqft,
#                 "lot_size": lot_size,
#             })

#     df = pd.DataFrame(records)
#     df = backfill_missing_fields_from_txt(df, get_txt_siblings(path))
#     return SalesCleaner.clean(df)


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

    suffix_map = {        "beresford": "Lord Roberts", "arnold": "Riverview", "wardlaw": "Osborne Village", "mcmillan": "Crescentwood",
        "fleet": "Osborne Village", "walker": "Lord Roberts", "ashland": "Riverview", "balfour": "Riverview",
        "montgomery": "Riverview", "morley": "Lord Roberts", "clark": "Riverview", "baltimore": "Riverview",
        "churchill": "Fort Rouge", "dudley": "Crescentwood", "mulvey": "Crescentwood", "scotland": "Crescentwood",
        "warsaw": "Crescentwood", "corydon": "Crescentwood", "hector": "Crescentwood", "gauvin": "St Boniface",
        "archibald": "St Boniface", "deniset": "St Boniface", "lariviere": "St Boniface", "lyndale": "Norwood Flats",
        "eugenie": "Norwood Flats", "redview": "St Vital", "kitson": "St Boniface", "dollard": "St Boniface",
        "langevin": "St Boniface"
    }

    address_regex = re.compile(r"\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place)")

    for page_text in pages:
        address_starts = [(m.start(), m.group()) for m in address_regex.finditer(page_text)]

        for i, (start, _) in enumerate(address_starts):
            end = address_starts[i + 1][0] if i + 1 < len(address_starts) else len(page_text)
            listing = page_text[start:end].strip()

            address_match = address_regex.search(listing)
            mls_match = re.search(r"MLS[\u00aeR#\s:]*?(\d{8})", listing)
            dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
            sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
            lot_match = re.search(r"(\d+\.\d+)\s*M2", listing)

            raw_addr = address_match.group().strip() if address_match else f"Unknown Address"
            address = clean_address_field(raw_addr)
            address_lower = address.lower()
            listing_lower = listing.lower()

            neighborhood = "Loose Area"
            for hood in known_neighborhoods:
                if hood.lower() in listing_lower or hood.lower() in address_lower:
                    neighborhood = hood
                    break

            if neighborhood == "Loose Area":
                for suffix, hood in suffix_map.items():
                    if suffix in address_lower:
                        neighborhood = hood
                        break

            mls_number = mls_match.group(1) if mls_match else hashlib.md5(address.encode()).hexdigest()[:10]
            dom = int(dom_match.group(1)) if dom_match else 0
            sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200
            lot_size = float(lot_match.group(1)) if lot_match else 0.0

            # Prices parsing per listing
            price_candidates = re.findall(r"\$\d{1,3}(?:,\d{3})*", listing)
            list_price, sold_price = 0.0, 0.0
            if len(price_candidates) >= 2:
                list_price = parse_currency(price_candidates[0])
                sold_price = parse_currency(price_candidates[1])

            if list_price == 0.0 or sold_price == 0.0:
                for txt in get_txt_siblings(path):
                    match = re.search(re.escape(address), txt, re.IGNORECASE)
                    if match:
                        before = txt[max(0, match.start()-300):match.start()]
                        after = txt[match.end():match.end()+300]
                        context = before + after
                        price_matches = re.findall(r"\$\d{1,3}(?:,\d{3})*", context)
                        if len(price_matches) >= 2:
                            list_price = parse_currency(price_matches[0])
                            sold_price = parse_currency(price_matches[1])
                        break

            garage_match = re.search(r"Parking[\s\S]{0,100}?((?:Detached|Attached|Pad|Plug-In|Rear|Carport|Drive Access|Parking Pad|Single|Double)[^\n]*)", listing, re.IGNORECASE)
            garage_raw = garage_match.group(1) if garage_match else ""
            garage_type = normalize_garage_type(garage_raw)

            style = match_known_terms(listing, HOUSE_STYLES)
            type_ = match_known_terms(listing, HOUSE_TYPES)
            if "Other" in type_ and len(type_.split(",")) > 1:
                type_ = ", ".join([t for t in type_.split(",") if t.strip() != "Other"])

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
                "style": style,
                "type": type_,
                "sqft": sqft,
                "lot_size": lot_size,
            })

    df = pd.DataFrame(records)
    df = backfill_missing_fields_from_txt(df, get_txt_siblings(path))
    return SalesCleaner.clean(df)
