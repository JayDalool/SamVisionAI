# file: tools/pdf_to_csv.py

import pdfplumber
import pandas as pd
import os
import re

OUTPUT_DIR = "parsed_csv"

def extract_csv_from_pdf(pdf_path):
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table:
                    cleaned = [col.strip() if col else "" for col in row]
                    if len(cleaned) < 5 or "Winnipeg" not in str(cleaned):
                        continue
                    rows.append(cleaned)
    
    if not rows:
        print(f"[SKIP] No rows parsed from {pdf_path}")
        return None

    df = pd.DataFrame(rows)
    return df


def convert_all_pdfs(input_folder="pdf_uploads"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_dfs = []

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            df = extract_csv_from_pdf(path)
            if df is not None:
                csv_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".csv"))
                df.to_csv(csv_path, index=False)
                all_dfs.append(df)
                print(f" Saved: {csv_path}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, "merged.csv"), index=False)
        print(" Merged CSV saved to: parsed_csv/merged.csv")
    else:
        print(" No data parsed from any PDF.")


if __name__ == "__main__":
    convert_all_pdfs()
