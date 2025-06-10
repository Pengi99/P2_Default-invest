"""Data collection utilities."""

import os
import pandas as pd
import numpy as np
import config # Assuming config.py is in the root and accessible

# Define number of sample companies and days for stock data
N_COMPANIES = 100
# N_DAYS_PER_YEAR = 252 # Approximate trading days. Using business days from date_range now.

def fetch_financial_data(year):
    """Fetch or generate sample financial statement data for a given year."""
    # TODO: Implement DART API calls for real data
    # For now, generate sample data
    stock_codes = [f"A{i:05d}" for i in range(N_COMPANIES)]
    data = {
        "stock_code": stock_codes,
        "year": year,
        "X1": np.random.normal(0.5, 0.2, N_COMPANIES),  # Example: Current Ratio
        "X2": np.random.normal(0.3, 0.1, N_COMPANIES),  # Example: Total Liabilities / Total Assets
        "X3": np.random.normal(0.1, 0.05, N_COMPANIES), # Example: Retained Earnings / Total Assets
        "X4": np.random.normal(1.5, 0.5, N_COMPANIES),  # Example: Sales / Total Assets
        "equity": np.random.lognormal(mean=20, sigma=1, size=N_COMPANIES), # In KRW
    }
    df = pd.DataFrame(data)
    # Ensure net_income is reasonable relative to equity for ROE calculation later
    df['net_income'] = df['equity'] * np.random.normal(loc=0.08, scale=0.15, size=N_COMPANIES) # Can be negative
    return df

def fetch_stock_data(year):
    """Fetch or generate sample stock price and market cap data for a given year."""
    # TODO: Implement stock data collection for real data
    # For now, generate sample data
    stock_codes = [f"A{i:05d}" for i in range(N_COMPANIES)]
    all_stock_data = []

    for stock_code in stock_codes:
        # Generate dates for business days within the year
        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='B')
        if dates.empty:
            continue # Skip if no business days (e.g., very short year range, though unlikely here)

        # Simulate stock prices with a random walk
        start_price = np.random.uniform(10000, 100000)
        # Generate returns for len(dates)-1 periods, then construct prices
        daily_returns_sim = np.random.normal(loc=0.0005, scale=0.02, size=len(dates)-1)
        prices = [start_price]
        for r in daily_returns_sim:
            prices.append(prices[-1] * (1 + r))
        
        prices = np.array(prices)
        prices[prices <= 0] = 0.01 # Ensure no negative prices

        stock_df = pd.DataFrame({
            "stock_code": stock_code,
            "date": dates,
            "close_price": prices,
            "year": year # Add year column for merging
        })
        
        stock_df["return"] = stock_df["close_price"].pct_change()
        stock_df["return"] = stock_df["return"].fillna(0) # Fill NaN for the first day's return

        stock_df["market_cap"] = stock_df["close_price"] * np.random.uniform(1e5, 1e7) 
        all_stock_data.append(stock_df)

    if not all_stock_data:
        return pd.DataFrame(columns=['stock_code', 'date', 'close_price', 'year', 'return', 'market_cap'])
        
    return pd.concat(all_stock_data, ignore_index=True)


def fetch_and_save_all_data(start_year, end_year):
    raw_data_path = os.path.join(config.DATA_PATH, "raw")
    os.makedirs(raw_data_path, exist_ok=True)
    for year_val in range(start_year, end_year + 1):
        print(f"Generating sample data for {year_val}...")
        fin_df = fetch_financial_data(year_val)
        stock_df = fetch_stock_data(year_val)

        if stock_df.empty or fin_df.empty:
            print(f"Skipping {year_val} due to empty stock or financial data.")
            continue

        # Merge financial data (annual) with stock data (daily)
        merged_df = pd.merge(stock_df, fin_df, on=["stock_code", "year"], how="left")
        
        output_path = os.path.join(raw_data_path, f"raw_data_{year_val}.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"Saved sample data for {year_val} to {output_path}")

if __name__ == '__main__':
    # This setup allows running the script directly from the src directory
    # It assumes config.py is in the parent directory of src/
    # For robust execution, ensure config can be imported (e.g. by setting PYTHONPATH or adjusting sys.path)
    # import sys
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # import config
    
    # Check if config is imported, if not, try to load it (basic attempt)
    if 'config' not in globals():
        import sys
        import os
        # Add project root to path to find config.py
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            import config
        except ImportError:
            print("Error: config.py not found. Make sure it's in the project root.")
            exit(1)

    fetch_and_save_all_data(config.START_YEAR, config.END_YEAR)
    print("Sample data generation complete.")
