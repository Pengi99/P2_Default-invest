"""Data collection utilities."""

import os
import pandas as pd


def fetch_financial_data(year):
    """Fetch financial statement data for a given year."""
    # TODO: Implement DART API calls
    return pd.DataFrame()


def fetch_stock_data(year):
    """Fetch stock price and market cap data for a given year."""
    # TODO: Implement stock data collection
    return pd.DataFrame()


def fetch_and_save_all_data(start_year, end_year):
    os.makedirs("data/raw", exist_ok=True)
    for year in range(start_year, end_year + 1):
        fin_df = fetch_financial_data(year)
        stock_df = fetch_stock_data(year)
        df = pd.concat([fin_df, stock_df], axis=1)
        df.to_csv(f"data/raw/raw_{year}.csv", index=False)
