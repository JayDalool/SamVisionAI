# file: scripts/parse_and_export.py
from utils.pdf_sales_parser import extract_pdf_sales
import os
import pandas as pd

PDF_DIR = "pdf_uploads"
OUT_CSV = "winnipeg_housing_data.csv"

def main():
    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not files:
        print("❌ No PDFs found to parse.")
        return

    all_data = []
    for file in files:
        pdf_path = os.path.join(PDF_DIR, file)
        df = extract_pdf_sales(pdf_path)
        if not df.empty:
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUT_CSV, index=False)
        print(f"✅ Exported {len(final_df)} rows to {OUT_CSV}")
    else:
        print("❌ No valid data parsed.")

if __name__ == "__main__":
    main()
