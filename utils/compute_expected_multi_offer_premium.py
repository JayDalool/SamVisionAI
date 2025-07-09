# utils/compute_expected_multi_offer_premium.py

import pandas as pd

def compute_expected_premiums(df, dom_cutoff=3, sell_list_ratio_cutoff=1.05):
    """
    Compute expected multi-offer premiums (%) per neighborhood.
    Multi-offer is defined as DOM <= 3 or Sell/List >= 1.05.
    """
    df = df.copy()
    df = df.dropna(subset=['sold_price', 'list_price', 'dom_days', 'neighborhood'])

    df['multi_offer_flag'] = (
        (df['dom_days'] <= dom_cutoff) |
        (df['sell_list_ratio'] >= sell_list_ratio_cutoff)
    ).astype(int)

    grouped = df.groupby(['neighborhood', 'multi_offer_flag'])['sold_price'].mean().reset_index()

    multi = grouped[grouped['multi_offer_flag'] == 1]
    on_time = grouped[grouped['multi_offer_flag'] == 0]

    premium_df = pd.merge(
        multi[['neighborhood', 'sold_price']],
        on_time[['neighborhood', 'sold_price']],
        on='neighborhood',
        suffixes=('_multi', '_on_time')
    )

    premium_df['expected_premium_pct'] = (
        (premium_df['sold_price_multi'] - premium_df['sold_price_on_time']) /
        premium_df['sold_price_on_time'] * 100
    ).round(1)

    premium_df = premium_df.sort_values(by='expected_premium_pct', ascending=False)

    return premium_df[['neighborhood', 'expected_premium_pct']]
