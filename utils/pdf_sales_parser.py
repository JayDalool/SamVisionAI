# # file: utils/pdf_sales_parser.py

# import fitz
# import re
# import pandas as pd
# import datetime
# import json
# from datetime import datetime
# from PyPDF2 import PdfReader
# from utils.pdf_text_utils import parse_pdf_text  # your utility to parse text from PDF
# from pdfminer.high_level import extract_text

# from typing import List, Dict
# from typing import Optional, Tuple, List, Callable, Any, Union

# REGION_FILE = "data/region_lookup.json"

# def get_season(date):
#     return ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Summer", "Summer", "Fall", "Fall", "Fall", "Winter"][date.month - 1]

# def safe_match(pattern: str, text: str, cast: Callable[[Any], Any] = str, group: Union[int, Tuple[int, ...]] = 1) -> Any:
#     match = re.search(pattern, text, re.IGNORECASE)
#     if not match:
#         return None
#     try:
#         if isinstance(group, tuple):
#             values = [match.group(g).replace(",", "").strip() for g in group]
#             return cast(values)
#         value = match.group(group).replace(",", "").strip()
#         return cast(value)
#     except:
#         return None

# def normalize_garage_type(raw: str) -> str:
#     raw = str(raw).lower()
#     mapping = {
#         "single_attached": ["single attached", "sgl att"],
#         "single_detached": ["single detached", "sgl det"],
#         "double_attached": ["double attached", "dbl att"],
#         "double_detached": ["double detached", "dbl det"],
#         "triple_attached": ["triple attached", "tpl att"],
#         "triple_detached": ["triple detached", "tpl det"],
#     }
#     for key, vals in mapping.items():
#         if any(v in raw for v in vals):
#             return key
#     return "none"

# def extract_text_from_pdf(pdf_path: str) -> str:
#     return extract_text(pdf_path)

# def extract_pdf_sales(pdf_path: str) -> pd.DataFrame:
#     text = extract_text_from_pdf(pdf_path)
#     blocks = re.split(r"(?=\nMLS.{0,3}#)", text, flags=re.IGNORECASE)
#     records = []

#     for block in blocks:
#         mls = extract_first_match(block, r"MLS.{0,3}#\s*(\d{9})", str)
#         address = extract_first_match(block, r"Address\s*:\s*(.+)", str)
#         neighborhood = extract_first_match(block, r"Area/Neighbr\s*:\s*(.+)", str)
#         list_price = extract_first_match(block, r"List Price\s*:\s*\$?([\d,]+)", lambda x: float(x.replace(",", "")), 1) or 0
#         sold_price = extract_first_match(block, r"Sold Price\s*:\s*\$?([\d,]+)", lambda x: float(x.replace(",", "")), 1) or 0
#         sell_list_ratio = extract_first_match(block, r"Sell/List Ratio\s*:\s*(\d+\.\d+)", float) or 0
#         dom = extract_first_match(block, r"DOM\s*:\s*(\d+)", int) or 0
#         bedrooms = extract_first_match(block, r"BDA:\s*(\d+)", int) or 0
#         bathrooms = extract_first_match(block, r"FB:\s*(\d+)\s+HB:\s*(\d+)", lambda x: int(x[0]) + 0.5 * int(x[1]), group=(1, 2)) or 1.0
#         garage_raw = extract_first_match(block, r"Parking\s*:\s*(.+)", str) or ""
#         garage_type = normalize_garage_type(garage_raw)
#         house_type = extract_first_match(block, r"Type\s*:\s*(.+)", str) or "none"
#         sqft = extract_first_match(block, r"Living Area\s*:\s*(\d{3,5})\s*SF", int) or 0
#         lot_frontage = extract_first_match(block, r"Frontage\s*:\s*(\d+)", int) or 0
#         lot_depth = extract_first_match(block, r"Depth\s*:\s*(\d+)", int) or 0
#         lot_size = lot_frontage * lot_depth if lot_frontage > 0 and lot_depth > 0 else 0

#         if mls and address:
#             records.append({
#                 "listing_date": datetime.today().date(),
#                 "season": get_current_season(),
#                 "mls_number": mls,
#                 "neighborhood": neighborhood or "none",
#                 "address": address,
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": sell_list_ratio,
#                 "dom": dom,
#                 "bedrooms": bedrooms,
#                 "bathrooms": bathrooms,
#                 "garage_type": garage_type,
#                 "house_type": house_type,
#                 "sqft": sqft,
#                 "lot_size": lot_size
#             })

#     return pd.DataFrame(records)


# def extract_first_match(text: str, pattern: str, cast: Callable = lambda x: x, group: Union[int, Tuple[int, ...]] = 1, default=None):
#     m = re.search(pattern, text, re.IGNORECASE)
#     if not m:
#         return default
#     try:
#         if isinstance(group, tuple):
#             return cast(tuple(m.group(i).strip() for i in group))
#         return cast(m.group(group).strip())
#     except Exception:
#         return default


# def get_current_season():
#     m = datetime.today().month
#     return ["Winter", "Spring", "Summer", "Fall"][(m % 12) // 3]

# def clean(s):
#     return s.replace('\n', ' ').strip()
# def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
#     try:
#         with open(REGION_FILE) as f:
#             region_map = json.load(f)
#     except:
#         region_map = {}

#     for i, row in df.iterrows():
#         if not row.get("season"):
#             df.at[i, "season"] = get_season(row.get("listing_date", datetime.today()))
#         if isinstance(row.get("garage_type"), str):
#             val = row["garage_type"].lower().replace("dbl", "double").replace("sgl", "single")
#             df.at[i, "garage_type"] = val.strip()
#         if row.get("sqft") and (row["sqft"] <= 0 or row["sqft"] > 10000):
#             df.at[i, "sqft"] = 0
#         if row.get("lot_size") and (row["lot_size"] <= 0 or row["lot_size"] > 100000):
#             df.at[i, "lot_size"] = 0
#         if isinstance(row.get("address"), str):
#             df.at[i, "address"] = row["address"].strip()
#             for key in region_map:
#                 if key.lower() in row["address"].lower():
#                     df.at[i, "region"] = region_map[key]
#                     break

#     df["sold_price"] = df["sold_price"].fillna(df["list_price"])
#     df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors='coerce').fillna(1.0)
#     for col in ["garage_type", "region", "neighborhood"]:
#         df[col] = df[col].fillna("none").astype(str)

#     return df
# import fitz
# import re
# import pandas as pd
# import datetime
# import json
# from datetime import datetime
# from PyPDF2 import PdfReader
# from utils.pdf_text_utils import parse_pdf_text  # your utility to parse text from PDF
# from pdfminer.high_level import extract_text

# from typing import List, Dict
# from typing import Optional, Tuple, List, Callable, Any, Union
# import re
# import pandas as pd
# from datetime import datetime
# from pdfminer.high_level import extract_text

# def extract_text_from_pdf(pdf_path: str) -> str:
#     return extract_text(pdf_path)

# def get_current_season():
#     return ["Winter", "Spring", "Summer", "Fall"][(datetime.today().month % 12) // 3]

# def extract_first_match(text: str, pattern: str, cast: Callable = lambda x: x, group: Union[int, Tuple[int, ...]] = 1, default=None):
#     m = re.search(pattern, text, re.IGNORECASE)
#     if not m:
#         return default
#     try:
#         if isinstance(group, tuple):
#             return cast(tuple(m.group(i).strip() for i in group))
#         return cast(m.group(group).strip())
#     except Exception:
#         return default

# def normalize_garage_type(raw: str) -> str:
#     raw = raw.lower()
#     if "single" in raw:
#         return "single_attached" if "attached" in raw else "single_detached"
#     if "double" in raw:
#         return "double_attached" if "attached" in raw else "double_detached"
#     if "triple" in raw:
#         return "triple_attached" if "attached" in raw else "triple_detached"
#     return "none"

# def extract_pdf_sales(pdf_path: str) -> pd.DataFrame:
#     text = extract_text_from_pdf(pdf_path)
#     blocks = re.split(r"\nSold\n", text, flags=re.IGNORECASE)
#     records = []

#     for block in blocks:
#         lines = block.strip().splitlines()
#         if len(lines) < 3:
#             continue

#         try:
#             mls = lines[0].strip()
#             area = lines[1].strip()
#             address = lines[2].strip() if len(lines) > 2 else "unknown"

#             list_price = extract_first_match(block, r"List Price\s*[:\-]?\s*\$?\s*([\d,]+)", float) or 0
#             sold_price = extract_first_match(block, r"Sold Price\s*[:\-]?\s*\$?\s*([\d,]+)", float) or 0
#             sell_list_ratio = extract_first_match(block, r"Sell/List Ratio\s*[:\-]?\s*([\d.]+)", float) or 0
#             dom = extract_first_match(block, r"DOM\s*[:\-]?\s*(\d+)", int) or 0

#             bedrooms = extract_first_match(block, r"BDA\s*[:\-]?\s*(\d+)", int) or 0
#             fb = extract_first_match(block, r"FB\s*[:\-]?\s*(\d+)", int) or 1
#             hb = extract_first_match(block, r"HB\s*[:\-]?\s*(\d+)", int) or 0
#             bathrooms = fb + 0.5 * hb

#             house_type = extract_first_match(block, r"Type\s*[:\-]?\s*(.+)", str) or "none"
#             sqft = extract_first_match(block, r"Living Area\s*[:\-]?\s*([\d,]+)", int) or 0

#             frontage = extract_first_match(block, r"Frontage\s*[:\-]?\s*([\d,]+)", int) or 0
#             depth = extract_first_match(block, r"Depth\s*[:\-]?\s*([\d,]+)", int) or 0
#             lot_size = frontage * depth if frontage > 0 and depth > 0 else 0

#             garage_raw = extract_first_match(block, r"Parking\s*[:\-]?\s*(.+)", str) or "none"
#             garage_type = normalize_garage_type(garage_raw)

#             records.append({
#                 "listing_date": datetime.today().date(),
#                 "season": get_current_season(),
#                 "mls_number": mls,
#                 "neighborhood": area,
#                 "address": address,
#                 "list_price": list_price,
#                 "sold_price": sold_price,
#                 "sell_list_ratio": sell_list_ratio,
#                 "dom": dom,
#                 "bedrooms": bedrooms,
#                 "bathrooms": bathrooms,
#                 "garage_type": garage_type,
#                 "house_type": house_type,
#                 "sqft": sqft,
#                 "lot_size": lot_size,
#             })
#         except Exception:
#             continue

#     return pd.DataFrame(records)

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

#     return pd.DataFrame(records)
import re
import pandas as pd
from datetime import datetime
from typing import Callable, Tuple, Union
import fitz  # PyMuPDF for reliable PDF text extraction

def extract_text_from_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]

def get_current_season():
    return ["Winter", "Spring", "Summer", "Fall"][(datetime.today().month % 12) // 3]

def parse_currency(value: str) -> float:
    try:
        return float(value.replace("$", "").replace(",", ""))
    except:
        return 0.0

def normalize_garage_type(raw: str) -> str:
    raw = raw.lower().strip()
    if not raw or raw == 'none':
        return "none"
    tags = []
    if "pad" in raw:
        tags.append("pad")
    if "plug" in raw:
        tags.append("plug-in")
    if "rear" in raw:
        tags.append("rear drive")
    if "unpaved" in raw:
        tags.append("unpaved")
    if "lot" in raw:
        tags.append("lot shape")
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

def extract_pdf_sales(pdf_path: str) -> pd.DataFrame:
    pages = extract_text_from_pdf(pdf_path)
    records = []

    for page_text in pages:
        listings = re.split(r"(?=MLS#[:\-]?)", page_text)
        for listing in listings:
            if not listing.strip():
                continue

            mls_match = re.search(r"MLS#\s*[:\-]?\s*(\w+)", listing)
            address_match = re.search(r"\n(\d{1,5}\s+[A-Za-z0-9 .,'\-]+(?:Avenue|Street|Drive|Road|Boulevard|Lane|Bay|Crescent|Place))", listing)
            if not address_match:
                address_match = re.search(r"Address\s*[:\-]?\s*(.*?)(?:\n|$)", listing)

            area_match = re.search(r"(?i)Area\s*[:\-]?\s*([^\n]+)", listing)
            dom_match = re.search(r"DOM\s*[:\-]?\s*(\d+)", listing, re.IGNORECASE)
            sqft_match = re.search(r"(\d{3,5})\s+SF", listing)
            sold_price_match = re.search(r"Sell Price\s*[:\-]?\s*\$([\d,]+)", listing)
            list_price_match = re.search(r"List Price\s*[:\-]?\s*\$([\d,]+)", listing)
            garage_match = re.search(r"Parking\s*[:\-]?\s*(.*?)\s*(?:\\n|\n|$)", listing, re.IGNORECASE)
            type_match = re.search(r"Type\s*[:\-]?\s*(.*?)\s*(?:\\n|\n|$)", listing, re.IGNORECASE)

            mls = mls_match.group(1) if mls_match else "unknown"
            raw_addr = address_match.group(1).strip() if address_match else f"Unknown Address #{mls}"
            address = clean_address_field(raw_addr)
            area = area_match.group(1).strip() if area_match else "Loose Area"
            dom = int(dom_match.group(1)) if dom_match else 0
            sqft = int(sqft_match.group(1).replace(",", "")) if sqft_match else 1200
            sold_price = parse_currency(sold_price_match.group(1)) if sold_price_match else 0.0
            list_price = parse_currency(list_price_match.group(1)) if list_price_match else 0.0
            garage_type = normalize_garage_type(garage_match.group(1)) if garage_match else "none"
            house_type = type_match.group(1).strip() if type_match else "house"

            records.append({
                "listing_date": datetime.today().date(),
                "season": get_current_season(),
                "mls_number": mls,
                "neighborhood": area,
                "address": address,
                "list_price": list_price,
                "sold_price": sold_price,
                "sell_list_ratio": round((sold_price / list_price), 2) if list_price else 0.0,
                "dom": dom,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "garage_type": garage_type,
                "house_type": house_type,
                "sqft": sqft,
                "lot_size": 0,
            })

    return pd.DataFrame(records)