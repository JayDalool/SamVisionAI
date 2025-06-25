
# file: utils/pdf_sales_parser.py

# import re
# import pandas as pd
# from datetime import datetime
# from pdfminer.high_level import extract_text
# from typing import Callable, Union, Tuple, Optional


# def extract_text_from_pdf(pdf_path: str) -> str:
#     return extract_text(pdf_path)


# def get_current_season():
#     m = datetime.today().month
#     return ["Winter", "Spring", "Summer", "Fall"][(m % 12) // 3]


# def parse_currency(value: str) -> float:
#     return float(value.replace("$", "").replace(",", ""))


# def extract_first_match(text: str, pattern: str, cast: Callable = lambda x: x,
#                         group: Union[int, Tuple[int, ...]] = 1, default=None):
#     m = re.search(pattern, text, re.IGNORECASE)
#     if not m:
#         return default
#     try:
#         if isinstance(group, tuple):
#             return cast(tuple(m.group(i).strip() for i in group))
#         return cast(m.group(group).strip())
#     except Exception:
#         return default


# def normalize_garage_type(raw: Optional[str]) -> str:
#     raw = raw.lower() if raw else ""
#     if "single" in raw:
#         return "single_attached" if "attached" in raw else "single_detached"
#     if "double" in raw:
#         return "double_attached" if "attached" in raw else "double_detached"
#     if "triple" in raw:
#         return "triple_attached" if "attached" in raw else "triple_detached"
#     return "none"


# def extract_house_blocks(text: str) -> list[dict]:
#     blocks = re.split(r"\nSold\n", text, flags=re.IGNORECASE)
#     parsed = []

#     for block in blocks:
#         lines = block.strip().splitlines()
#         if len(lines) < 3:
#             continue

#         mls = lines[0].strip()
#         area = lines[1].strip()
#         address = lines[2].strip()

#         house_type = extract_first_match(block, r"Type\s*[:\-]?\s*(.+)", str, default="none")
#         sqft = extract_first_match(block, r"Living Area\s*[:\-]?\s*([\d,]+)", lambda x: int(x.replace(",", "")), default=0)
#         frontage = extract_first_match(block, r"Frontage\s*[:\-]?\s*([\d,]+)", lambda x: float(x.replace(",", "")), default=0)
#         depth = extract_first_match(block, r"Depth\s*[:\-]?\s*([\d,]+)", lambda x: float(x.replace(",", "")), default=0)
#         lot_size = frontage * depth
#         garage_raw = extract_first_match(block, r"Parking\s*[:\-]?\s*(.+)", str, default=None)
#         garage_type = normalize_garage_type(garage_raw)

#         parsed.append({
#             "mls_number": mls,
#             "neighborhood": area,
#             "address": address,
#             "house_type": house_type,
#             "garage_type": garage_type,
#             "sqft": sqft,
#             "lot_size": lot_size
#         })
#     return parsed


# def extract_sales_rows(text: str) -> list[dict]:
#     pattern = re.compile(
#         r"\$(\d{1,3}(?:,\d{3})+)\s+"
#         r"\$(\d{1,3}(?:,\d{3})+)\s+"
#         r"(\d{2,3}(?:\.\d+)?)%\s+"
#         r"(\d+)\s+"
#         r"(\d+)\s+"
#         r"(\d+)\s+"
#         r"(\d+)\s+"
#         r"([\d,]+)\s+SF",
#         re.IGNORECASE
#     )
#     matches = pattern.findall(text)
#     sales = []
#     for m in matches:
#         try:
#             list_price = parse_currency(m[0])
#             sold_price = parse_currency(m[1])
#             sell_list_ratio = float(m[2]) / 100.0
#             dom = int(m[3])
#             bedrooms = int(m[4])
#             fb = int(m[5])
#             hb = int(m[6])
#             bathrooms = fb + 0.5 * hb
#             sqft = int(m[7].replace(",", ""))

#             sales.append({
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": sell_list_ratio,
#                 "dom": dom,
#                 "bedrooms": bedrooms,
#                 "bathrooms": bathrooms,
#                 "sqft": sqft
#             })
#         except Exception:
#             continue
#     return sales


# def extract_pdf_sales(pdf_path: str) -> pd.DataFrame:
#     text = extract_text_from_pdf(pdf_path)
#     house_blocks = extract_house_blocks(text)
#     sales_data = extract_sales_rows(text)

#     records = []
#     for i in range(min(len(house_blocks), len(sales_data))):
#         house = house_blocks[i]
#         sale = sales_data[i]
#         records.append({
#             "listing_date": datetime.today().date(),
#             "season": get_current_season(),
#             **house,
#             **sale
#         })

# #     return pd.DataFrame(records)
# import re
# import pandas as pd
# from datetime import datetime
# from typing import Callable, Tuple, Union
# import fitz  # PyMuPDF for reliable PDF text extraction

# def extract_text_from_pdf(pdf_path: str) -> list:
#     doc = fitz.open(pdf_path)
#     return [page.get_text() for page in doc]

# def get_current_season():
#     return ["Winter", "Spring", "Summer", "Fall"][(datetime.today().month % 12) // 3]

# def parse_currency(value: str) -> float:
#     try:
#         return float(value.replace("$", "").replace(",", ""))
#     except:
#         return 0.0

# def normalize_garage_type(raw: str) -> str:
#     raw = raw.lower().strip()
#     if not raw or raw == 'none':
#         return "none"
#     tags = []
#     if "pad" in raw:
#         tags.append("pad")
#     if "plug" in raw:
#         tags.append("plug-in")
#     if "rear" in raw:
#         tags.append("rear drive")
#     if "unpaved" in raw:
#         tags.append("unpaved")
#     if "lot" in raw:
#         tags.append("lot shape")
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

# def extract_pdf_sales(pdf_path: str) -> pd.DataFrame:
#     pages = extract_text_from_pdf(pdf_path)
#     records = []

#     for page_text in pages:
#         listings = re.split(r"(?=MLS#[:\-]?)", page_text)
#         for listing in listings:
#             if not listing.strip():
#                 continue

#             mls_match = re.search(r"MLS#\s*[:\-]?\s*(\w+)", listing)
#             address_match = re.search(r"\n(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", listing)
#             if not address_match:
#                 address_match = re.search(r"Address\s*[:\-]?\s*(.*?)(?:\n|$)", listing)

#             area_match = re.search(r"(?i)Area\s*[:\-]?\s*([^\n]+)", listing)
#             dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
#             sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
#             sold_price_match = re.search(r"Sell Price\s*[:\-]?\s*\$([\d,]+)", listing)
#             list_price_match = re.search(r"List Price\s*[:\-]?\s*\$([\d,]+)", listing)
#             garage_match = re.search(r"Parking\s*[:\-]?\s*(.*?)\s*(?:\\n|\n|$)", listing, re.IGNORECASE)
#             type_match = re.search(r"Type\s*[:\-]?\s*(.*?)\s*(?:\\n|\n|$)", listing, re.IGNORECASE)

#             mls = mls_match.group(1) if mls_match else "unknown"
#             raw_addr = address_match.group(1).strip() if address_match else f"Unknown Address #{mls}"
#             address = clean_address_field(raw_addr)
#             area = area_match.group(1).strip() if area_match else "Loose Area"
#             dom = int(dom_match.group(1)) if dom_match else 0
#             sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200
#             sold_price = parse_currency(sold_price_match.group(1)) if sold_price_match else 0.0
#             list_price = parse_currency(list_price_match.group(1)) if list_price_match else 0.0
#             garage_type = normalize_garage_type(garage_match.group(1)) if garage_match else "none"
#             house_type = type_match.group(1).strip() if type_match else "house"

#             records.append({
#                 "listing_date": datetime.today().date(),
#                 "season": get_current_season(),
#                 "mls_number": mls,
#                 "neighborhood": area,
#                 "address": address,
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
#                 "dom": dom,
#                 "bedrooms": 3,
#                 "bathrooms": 2.0,
#                 "garage_type": garage_type,
#                 "house_type": house_type,
#                 "sqft": sqft,
#                 "lot_size": 0,
#             })

#     return pd.DataFrame(records)

# file: utils/pdf_sales_parser.py

# file: utils/pdf_sales_parser.py

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
#         tags.append("pad")
#     if "plug" in raw:
#         tags.append("plug-in")
#     if "rear" in raw:
#         tags.append("rear drive")
#     if "unpaved" in raw:
#         tags.append("unpaved")
#     if "lot" in raw:
#         tags.append("lot shape")
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

#     for page_text in pages:
#         entries = re.split(r"(?=Sold\d{8})", page_text)
#         for listing in entries:
#             if not listing.strip():
#                 continue

#             address_match = re.search(r"(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", listing)
#             area_match = re.search(r"(?i)(?:Area/Neighbr|A/)[\s\:]*([\w\-/ ]+)", listing)
#             dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
#             sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
#             price_matches = re.findall(r"\$(\d{1,3}(?:,\d{3})*)", listing)
#             garage_match = re.search(r"Parking\s*[:\-]?\s*(.*?)\s*(?:\\n|\n|$)", listing, re.IGNORECASE)
#             type_match = re.search(r"Type\s*[:\-]?\s*(.*?)\s*(?:\\n|\n|$)", listing, re.IGNORECASE)
#             mls_match = re.search(r"Sold(\d{8})", listing)

#             raw_addr = address_match.group(1).strip() if address_match else f"Unknown Address"
#             address = clean_address_field(raw_addr)
#             area = area_match.group(1).strip() if area_match else "Loose Area"
#             dom = int(dom_match.group(1)) if dom_match else 0
#             sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200
#             list_price = parse_currency(price_matches[-4]) if len(price_matches) >= 4 else 0.0
#             sold_price = parse_currency(price_matches[-2]) if len(price_matches) >= 4 else 0.0
#             garage_type = normalize_garage_type(garage_match.group(1)) if garage_match else "none"
#             house_type = type_match.group(1).strip() if type_match else "Single Family Detached"
#             mls_number = mls_match.group(1) if mls_match else "unknown"

#             bedrooms = parse_bedrooms(listing)
#             bathrooms = parse_bathrooms(listing)

#             records.append({
#                 "listing_date": datetime.today().date(),
#                 "season": get_current_season(),
#                 "mls_number": mls_number,
#                 "neighborhood": area,
#                 "address": address,
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
#                 "dom": dom,
#                 "bedrooms": bedrooms,
#                 "bathrooms": bathrooms,
#                 "garage_type": garage_type,
#                 "house_type": house_type,
#                 "sqft": sqft,
#                 "lot_size": 0,
#             })

#     return pd.DataFrame(records)
# file: utils/pdf_sales_parser.py
# file: utils/pdf_sales_parser.py

# import re
# import pandas as pd
# from datetime import datetime
# from typing import List
# from pdfminer.high_level import extract_text
# import os
# import geopy
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter

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
#         tags.append("pad")
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

# geolocator = Nominatim(user_agent="real_estate_parser")
# geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# def get_neighborhood_from_geocode(address: str) -> str:
#     try:
#         location = geocode(f"{address}, Winnipeg, MB")
#         if location and hasattr(location, 'raw'):
#             return location.raw.get('address', {}).get('suburb') or "Loose Area"
#     except:
#         pass
#     return "Loose Area"

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

#             listing_lower = listing.lower()
#             address_lower = address.lower()

#             neighborhood = "Loose Area"
#             for hood in known_neighborhoods:
#                 hood_lower = hood.lower()
#                 if hood_lower in listing_lower or hood_lower in address_lower:
#                     neighborhood = hood
#                     break

#             if neighborhood == "Loose Area":
#                 address_suffix = re.search(r"\d{1,5}\s+(.*?)$", address)
#                 if address_suffix:
#                     for hood in known_neighborhoods:
#                         if hood.lower() in address_suffix.group(1).lower():
#                             neighborhood = hood
#                             break

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
#                     "baltimore": "Riverview"
#                 }
#                 for suffix, hood in suffix_map.items():
#                     if suffix in address_lower:
#                         neighborhood = hood
#                         break

#             if neighborhood == "Loose Area":
#                 neighborhood = get_neighborhood_from_geocode(address)

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
#                 "sqft": sqft,
#                 "lot_size": lot_size,
#             })

#     return pd.DataFrame(records)

# file: utils/pdf_sales_parser.py

import re
import pandas as pd
from datetime import datetime
from typing import List
from pdfminer.high_level import extract_text
import os

def extract_text_from_path(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = extract_text(path)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Only .pdf and .txt are supported.")
    return text.split("Page ")
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
        # Split by sets of 3 properties (each starts with Sold status line)
        property_blocks = re.split(r"(?=Status\s+Sold\s+Sold\s+Sold)", page_text)
        
        for block in property_blocks:
            if not block.strip():
                continue
                
            # Split each property block into 3 individual properties
            properties = re.split(r"(?=\S+\s+\S+\s+\S+)", block)
            
            for prop in properties[1:4]:  # first element is empty
                if not prop.strip():
                    continue
                
                try:
                    # MLS Number
                    mls_match = re.search(r"MLS[\u00aeR#\s:]*?(\d{9})", prop)
                    mls_number = mls_match.group(1) if mls_match else "unknown"
                    
                    # Neighborhood
                    neigh_match = re.search(r"Area/Neighbr\s+([^\n]+)", prop)
                    neighborhood = "Loose Area"
                    if neigh_match:
                        neighborhood_part = neigh_match.group(1).strip()
                        for hood in known_neighborhoods:
                            if hood.lower() in neighborhood_part.lower():
                                neighborhood = hood
                                break
                    
                    # Address
                    addr_match = re.search(r"Address\s+([^\n]+)", prop)
                    address = addr_match.group(1).strip() if addr_match else "Unknown Address"
                    address = clean_address_field(address)
                    
                    # House Type
                    type_match = re.search(r"Type\s+([^\n]+)", prop)
                    house_type = type_match.group(1).strip() if type_match else "Single Family Detached"
                    
                    # Style
                    style_match = re.search(r"Style\s+([^\n]+)", prop)
                    style = style_match.group(1).strip() if style_match else ""
                    
                    # Square Footage
                    sqft_match = re.search(r"Living Area\s+(\d+)\s*SF", prop)
                    sqft = int(sqft_match.group(1)) if sqft_match else 1200
                    
                    # Lot Size
                    lot_match = re.search(r"(\d+\.\d+)\s*M2", prop)
                    lot_size = float(lot_match.group(1)) if lot_match else 0.0
                    
                    # Prices
                    price_matches = re.findall(r"\$([\d,]+)", prop)
                    list_price = parse_currency(price_matches[-6]) if len(price_matches) >= 6 else 0.0
                    sold_price = parse_currency(price_matches[-3]) if len(price_matches) >= 3 else 0.0
                    
                    # DOM
                    dom_match = re.search(r"DOM\s*(\d+)", prop)
                    dom = int(dom_match.group(1)) if dom_match else 0
                    
                    # Bedrooms/Bathrooms
                    bedrooms = parse_bedrooms(prop)
                    bathrooms = parse_bathrooms(prop)
                    
                    # Garage Type
                    garage_raw = ""
                    parking_section = re.search(r"Parking([\s\S]+?)(?=\n\S)", prop)
                    if parking_section:
                        garage_raw = parking_section.group(1)
                    garage_type = normalize_garage_type(garage_raw)
                    
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
                        "style": style,
                    })
                
                except Exception as e:
                    print(f"Error parsing property: {e}")
                    continue

    return pd.DataFrame(records)