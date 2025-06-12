# data/clean_and_validate_csv.py

import os
import pandas as pd

INPUT = "parsed_csv/merged.csv"
OUTPUT = "parsed_csv/validated.csv"

if not os.path.exists(INPUT):
    print(f"❌ Error: File not found - {INPUT}")
    exit(1)

try:
    df = pd.read_csv(INPUT)
except pd.errors.EmptyDataError:
    print(f"❌ Error: Empty CSV - {INPUT}")
    exit(1)

# Optional: perform additional cleaning if needed
# Example: drop duplicates, normalize types, or rename columns
# df.drop_duplicates(inplace=True)

# Write validated file
df.to_csv(OUTPUT, index=False)
print(f"✅ Saved validated CSV: {OUTPUT}")
