# file: utils/pdf_sales_parser.py

import fitz  # PyMuPDF
import re
import pandas as pd
import datetime
import psycopg2
from typing import List, Tuple


class StructuredSalesPDFParser:
    """Parses real estate sales PDF files with structured text layout."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.records = []

    def parse(self) -> pd.DataFrame:
        doc = fitz.open(self.pdf_path)
        for page in doc:
            try:
                blocks = [block[4].strip().replace("\n", " ") for block in page.get_text("blocks")]
            except Exception as e:
                print(f"Error reading page text: {e}")
                continue

            for i in range(len(blocks) - 6):
                l1, l2, l3, l4, l5, l6, l7 = blocks[i:i+7]

                if "Price:" in l1 and "DOM" in l2 and "Neighbr:" in l3:
                    try:
                        record = {
                            "address": l1.split("Price:")[0].strip(),
                            "sold_price": int(re.search(r"Price:\s*\$([\d,]+)", l1).group(1).replace(",", "")) if re.search(r"Price:\s*\$([\d,]+)", l1) else None,
                            "mls_id": re.search(r"MLS\S*\s*#?:?\s*(\d+)", l2).group(1) if re.search(r"MLS\S*\s*#?:?\s*(\d+)", l2) else None,
                            "dom": int(re.search(r"DOM[:\s]*(\d+)", l2).group(1)) if re.search(r"DOM[:\s]*(\d+)", l2) else None,
                            "neighborhood": re.search(r"Neighbr:\s*(.*?)\s", l3).group(1) if re.search(r"Neighbr:\s*(.*?)\s", l3) else None,
                            "built_year": int(re.search(r"Yr Blt/Age:\s*(\d{4})", l3).group(1)) if re.search(r"Yr Blt/Age:\s*(\d{4})", l3) else None,
                            "house_type": l4.split("Style:")[0].replace("Type:", "").strip(),
                            "bedrooms": int(re.search(r"Beds:\s*BD(\d+)", l5).group(1)) if re.search(r"Beds:\s*BD(\d+)", l5) else None,
                            "bathrooms": (
                                int(re.search(r"FB:(\d+)", l6).group(1)) +
                                0.5 * int(re.search(r"HB:\s*(\d+)", l6).group(1))
                            ) if re.search(r"FB:(\d+)", l6) and re.search(r"HB:\s*(\d+)", l6) else None,
                            "garage_type": l7.replace("Parking:", "").strip()
                        }
                        self.records.append(record)
                    except Exception:
                        continue

        return pd.DataFrame(self.records)


def insert_sales_records_to_db(df: pd.DataFrame, db_config: dict) -> Tuple[int, List[str]]:
    inserted, skipped = 0, []
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        try:
            cursor.execute('''
                INSERT INTO housing_data (
                    neighborhood, house_type, bedrooms, bathrooms, built_year,
                    garage_type, listing_date, season, sold_price, dom, address
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                row.get("neighborhood"), row.get("house_type"),
                row.get("bedrooms"), row.get("bathrooms"), row.get("built_year"),
                row.get("garage_type"), datetime.date.today(), "Spring",
                row.get("sold_price"), row.get("dom"), row.get("address")
            ))
            inserted += 1
        except:
            skipped.append(row.get("address"))

    conn.commit()
    cursor.close()
    conn.close()
    return inserted, skipped
