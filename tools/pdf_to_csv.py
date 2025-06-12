# file: tools/pdf_to_csv.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import pandas as pd
from utils.pdf_sales_parser import extract_pdf_sales

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PDF_DIR = "pdf_uploads/"
OUTPUT_CSV = "parsed_csv/merged.csv"

def convert_all_pdfs():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_data = []
    for filename in sorted(os.listdir(PDF_DIR)):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"> Parsing: {filename}")
            try:
                df = extract_pdf_sales(pdf_path)
                if not df.empty:
                    print(f"✅ Parsed {len(df)} rows from {filename}")
                    all_data.append(df)
                else:
                    print(f"[SKIP] Empty DataFrame from {filename}")
            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Saved merged CSV: {OUTPUT_CSV}")
    else:
        print("⚠️ No valid data parsed. No file written.")

if __name__ == "__main__":
    convert_all_pdfs()
