"""Data preprocessing utilities."""

import os
import pandas as pd
import numpy as np
import config


def run_preprocessing():
    df = load_and_combine_data()
    df = clean_data(df)
    df = calculate_features_and_labels(df)
    split_data_chronologically(df)


def load_and_combine_data():
    frames = []
    for file in sorted(os.listdir("data/raw")):
        if file.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join("data/raw", file)))
    return pd.concat(frames, ignore_index=True)


def clean_data(df):
    """Basic data cleaning."""
    # Convert 'date' to datetime objects
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Fill NaNs in 'return' column with 0
    # This assumes that if a return is NaN, it's equivalent to a 0% return for that period.
    if 'return' in df.columns:
        df['return'] = df['return'].fillna(0)

    # TODO: Implement more sophisticated cleaning for real-world data
    # Examples:
    # - Outlier detection and treatment for financial ratios
    # - More advanced imputation strategies for missing values (beyond df.get(col, default))
    # - Handling of delisted stocks or M&A events if relevant to 'return'

    # Drop rows with any remaining NaNs.
    # For K-1 and quality factor, .get(col, default) handles NaNs at calculation time.
    # This dropna primarily targets other unexpected NaNs (e.g. in stock_code, year).
    df = df.dropna() # Consider subset=['stock_code', 'year', 'date'] for critical identifiers
    return df


def calculate_features_and_labels(df):
    # Calculate K-1 score
    df["k1_score"] = (
        6.56 * df.get("X1", 0)  # Using .get for robustness against missing columns
        + 3.26 * df.get("X2", 0)
        + 6.72 * df.get("X3", 0)
        + 1.05 * df.get("X4", 0)
    )
    df["bankrupt_label"] = (df["k1_score"] < config.K1_SCORE_THRESHOLD).astype(int)
    print(f"Bankrupt label distribution in combined data:\n{df['bankrupt_label'].value_counts(normalize=True)}")

    if config.QUALITY_FACTOR == 'ROE':
        net_income = df.get("net_income") # Returns None if column missing
        equity = df.get("equity")       # Returns None if column missing

        if net_income is not None and equity is not None:
            # Ensure net_income and equity are numeric before division
            net_income_numeric = pd.to_numeric(net_income, errors='coerce')
            equity_numeric = pd.to_numeric(equity, errors='coerce')
            
            # Calculate ROE where equity is positive; otherwise, NaN
            df["quality_factor"] = np.where(equity_numeric > 0, net_income_numeric / equity_numeric, np.nan)
            
            # Handle potential inf values if any (e.g. if equity_numeric was extremely small but positive)
            df["quality_factor"] = df["quality_factor"].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaNs in quality_factor (e.g., from division by zero/negative equity, or non-numeric) with 0
            # This implies a neutral or low quality if ROE is problematic or undefined.
            df["quality_factor"] = df["quality_factor"].fillna(0)
        else:
            # If 'net_income' or 'equity' columns are missing, default 'quality_factor' to 0
            df["quality_factor"] = 0
            
    elif config.QUALITY_FACTOR == 'SOME_OTHER_FACTOR': # Example for future extension
        # df["quality_factor"] = ... # calculation for another factor
        pass # Implement other quality factors as needed
    else:
        # Default quality factor if not specified or not implemented
        df["quality_factor"] = 0 
    return df


def split_data_chronologically(df):
    os.makedirs("data/processed", exist_ok=True)
    train = df[df["year"] <= config.TRAIN_END_YEAR]
    val = df[(df["year"] > config.TRAIN_END_YEAR) & (df["year"] <= config.VALIDATION_END_YEAR)]
    test = df[df["year"] > config.VALIDATION_END_YEAR]
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
