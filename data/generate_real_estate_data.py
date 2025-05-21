# file: data/generate_real_estate_data.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Define constants
NEIGHBORHOODS = {
    'Transcona': ('East', 400_000, 49.8954, -97.0153),
    'St. Vital': ('South', 500_000, 49.8200, -97.0900),
    'Fort Garry': ('Southwest', 600_000, 49.8384, -97.1664),
    'River Heights': ('Central', 700_000, 49.8667, -97.2000),
    'Tuxedo': ('West', 900_000, 49.8700, -97.2100),
    'Bridgewater': ('Southwest', 750_000, 49.7917, -97.1875),
    'Prairie Pointe': ('Southwest', 720_000, 49.7925, -97.1950),
    'East Kildonan': ('Northeast', 450_000, 49.9267, -97.0650),
    'West End': ('West', 480_000, 49.8920, -97.1650),
    'North Kildonan': ('North', 460_000, 49.9442, -97.0422),
    'St. James': ('West', 470_000, 49.8900, -97.2000)
}

HOUSE_TYPES = {
    'Detached': 470_399,
    'Semi-Detached': 400_000,
    'Townhouse': 375_785,
    'Bungalow': 460_000,
    'Split-Level': 450_000,
    'Condominium': 277_068,
    'Exoubekee': 500_000
}

SEASONS = ['Winter', 'Spring', 'Summer', 'Fall']

# Generate synthetic housing data
def generate_data(n_samples: int = 2000) -> pd.DataFrame:
    data = []
    base_date = datetime(2022, 1, 1)

    for _ in range(n_samples):
        neighborhood, (region, base_price, lat, lon) = random.choice(list(NEIGHBORHOODS.items()))
        house_type, type_price = random.choice(list(HOUSE_TYPES.items()))
        bedrooms = random.randint(2, 5)
        bathrooms = random.randint(1, 3)
        sqft = random.randint(800, 3000)
        lot_size = round(random.uniform(2.0, 10.0), 2)
        age = random.randint(1, 60)
        garage = random.choice([True, False])

        days_offset = random.randint(0, 730)
        listing_date = base_date + timedelta(days=days_offset)
        month = listing_date.month
        season = SEASONS[(month % 12) // 3]

        # Price calculation
        price = base_price
        price += (sqft - 1500) * 100
        price += (bedrooms - 3) * 5000
        price += (bathrooms - 2) * 3000
        price += (type_price - 402_915)
        price -= age * 1000
        if garage:
            price += 15000

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
            'age': age,
            'garage': garage,
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
    print("Data generated and saved to 'winnipeg_housing_data.csv'")

