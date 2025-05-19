# file: data/load_to_postgresql.py

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import ast

# Load DB config
with open("config.txt", "r") as file:
    db_config = ast.literal_eval(file.read())

# Connect
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Read updated CSV
df = pd.read_csv('winnipeg_housing_data.csv')

# Create table (matches new schema)
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
    listing_date DATE,
    season VARCHAR(20),
    latitude NUMERIC(9,6),
    longitude NUMERIC(9,6),
    sold_price INTEGER
)
''')

# Columns to insert
columns = [
    'neighborhood', 'region', 'house_type', 'bedrooms', 'bathrooms', 'sqft',
    'lot_size', 'built_year', 'garage_type', 'listing_date',
    'season', 'latitude', 'longitude', 'sold_price'
]

# Data prep
data_tuples = [tuple(x) for x in df[columns].to_numpy()]

# Insert
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
