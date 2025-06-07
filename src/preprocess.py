"""Data preprocessing utilities."""

import os
import pandas as pd
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
    # TODO: implement cleaning
    return df.dropna()


def calculate_features_and_labels(df):
    # Calculate K-1 score
    df["k1_score"] = (
        6.56 * df.get("X1", 0)
        + 3.26 * df.get("X2", 0)
        + 6.72 * df.get("X3", 0)
        + 1.05 * df.get("X4", 0)
    )
    df["bankrupt_label"] = (df["k1_score"] < config.K1_SCORE_THRESHOLD).astype(int)
    if config.QUALITY_FACTOR == 'ROE':
        df["quality_factor"] = df.get("net_income", 0) / df.get("equity", 1)
    return df


def split_data_chronologically(df):
    os.makedirs("data/processed", exist_ok=True)
    train = df[df["year"] <= config.TRAIN_END_YEAR]
    val = df[(df["year"] > config.TRAIN_END_YEAR) & (df["year"] <= config.VALIDATION_END_YEAR)]
    test = df[df["year"] > config.VALIDATION_END_YEAR]
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
