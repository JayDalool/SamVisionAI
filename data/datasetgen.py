# file: data/generate_real_estate_data.py

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta

# Load prices from JSON
with open("data/price_config.json", "r") as f:
    HOUSE_TYPES = json.load(f)

with open("data/region_prices.json", "r") as f:
    NEIGHBORHOODS = json.load(f)

SEASONS = ['Winter', 'Spring', 'Summer', 'Fall']
CURRENT_YEAR = datetime.now().year

GARAGE_TYPES = [
    'Detached 1-Car', 'Detached 2-Car', 'Detached 3-Car',
    'Attached 1-Car', 'Attached 2-Car', 'Attached 3-Car',
    'Attached 4-Car', 'Detached 4-Car', 'Detached 5-Car',
    'Attached 5-Car', 'Attached 6-Car'
]

def generate_data(n_samples: int = 2000) -> pd.DataFrame:
    data = []
    base_date = datetime(2022, 1, 1)
    current_month = datetime.now().strftime('%B')

    for _ in range(n_samples):
        n_data = random.choice(list(NEIGHBORHOODS.values()))
        base_price = n_data["monthly_prices"].get(current_month, n_data["default_price"])

        neighborhood = n_data["name"]
        region = n_data["region"]
        lat = n_data["latitude"]
        lon = n_data["longitude"]

        house_type = random.choice(list(HOUSE_TYPES.keys()))
        type_price = HOUSE_TYPES[house_type]

        bedrooms = random.randint(2, 5)
        bathrooms = random.randint(1, 3)
        sqft = random.randint(800, 3000)
        lot_size = round(random.uniform(2000, 10000), 2)

        built_year = random.randint(1965, CURRENT_YEAR)
        age = CURRENT_YEAR - built_year

        garage_type = random.choice(GARAGE_TYPES)
        garage_cars = int(garage_type.split()[1].replace("-Car", ""))
        garage_bonus = garage_cars * 4000 + (5000 if "Attached" in garage_type else 2000)

        days_offset = random.randint(0, 730)
        listing_date = base_date + timedelta(days=days_offset)
        month = listing_date.month
        season = SEASONS[(month % 12) // 3]

        price = base_price
        price += (sqft - 1500) * 100
        price += (bedrooms - 3) * 5000
        price += (bathrooms - 2) * 3000
        price += (type_price - 402_915)
        price -= age * 1000
        price += garage_bonus
        price += np.random.normal(0, 10000)

        sold_price = round(price, -3)

        data.append({
            'neighborhood': neighborhood,
            'region': region,
            'house_type': house_type,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft': sqft,
            'lot_size': lot_size,
            'built_year': built_year,
            'garage_type': garage_type,
            'listing_date': listing_date.date(),
            'season': season,
            'latitude': lat,
            'longitude': lon,
            'sold_price': sold_price
        })

    return pd.DataFrame(data)

if __name__ == '__main__':
    df = generate_data(3000)
    df.to_csv('winnipeg_housing_data.csv', index=False)
    print("✅ Data generated and saved to 'winnipeg_housing_data.csv'")
