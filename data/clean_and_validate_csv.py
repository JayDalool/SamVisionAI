# file: data/clean_and_validate_csv.py

import pandas as pd
import os

INPUT = "parsed_csv/merged.csv"
OUTPUT = "parsed_csv/validated.csv"

if not os.path.exists(INPUT):
    raise FileNotFoundError(f"{INPUT} not found")

df = pd.read_csv(INPUT)

# Drop rows with missing crucial fields
df_clean = df.dropna(subset=["address", "sold_price"])

# Optional: filter out junk/zeros
df_clean = df_clean[df_clean["sold_price"] > 1000]

df_clean.to_csv(OUTPUT, index=False)
print(f"✅ Cleaned & validated rows saved to {OUTPUT}")
