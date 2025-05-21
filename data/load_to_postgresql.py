# file: data/load_to_postgresql.py

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import ast

# Read and safely parse the config as a Python dictionary
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

# Connect using unpacked config
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Read CSV
csv_path = 'winnipeg_housing_data.csv'
df = pd.read_csv(csv_path)

# Convert garage column to boolean
if 'garage' in df.columns:
    df['garage'] = df['garage'].astype(bool)

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS housing_data (
    id SERIAL PRIMARY KEY,
    neighborhood VARCHAR(50),
    region VARCHAR(50),
    house_type VARCHAR(50),
    bedrooms INTEGER,
    bathrooms INTEGER,
    sqft INTEGER,
    lot_size NUMERIC(7,2),
    built_year INTEGER,
    garage_type VARCHAR(50),
    address VARCHAR(100),
    dom INTEGER,
    listing_date DATE,
    season VARCHAR(20),
    latitude NUMERIC(9,6),
    longitude NUMERIC(9,6),
    sold_price INTEGER
)

''')

# Prepare data for insertion
columns = [
    'neighborhood', 'region', 'house_type', 'bedrooms', 'bathrooms', 'sqft',
    'lot_size', 'built_year', 'garage_type', 'listing_date',
    'season', 'latitude', 'longitude', 'sold_price'
]

data_tuples = [tuple(x) for x in df[columns].to_numpy()]

# Bulk insert
insert_query = f'''
INSERT INTO housing_data (
    {', '.join(columns)}
) VALUES %s
'''

execute_values(cursor, insert_query, data_tuples)
conn.commit()

cursor.close()
conn.close()
print("✅ Data loaded into PostgreSQL table 'housing_data'")
