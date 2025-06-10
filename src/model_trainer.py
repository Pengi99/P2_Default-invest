"""Model training utilities."""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
import config


def run_training_pipeline():
    X_train, y_train, X_val, y_val = load_split_data()
    model = train_and_evaluate_models(X_train, y_train, X_val, y_val)
    save_model(model)
    return model


def load_split_data():
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, "processed", "train.csv"))
    val_df = pd.read_csv(os.path.join(config.DATA_PATH, "processed", "val.csv"))

    # Ensure all feature columns are present
    missing_train_cols = [col for col in config.FEATURE_COLUMNS if col not in train_df.columns]
    if missing_train_cols:
        raise ValueError(f"Feature columns {missing_train_cols} not found in train.csv")
    
    missing_val_cols = [col for col in config.FEATURE_COLUMNS if col not in val_df.columns]
    if missing_val_cols:
        raise ValueError(f"Feature columns {missing_val_cols} not found in val.csv")

    X_train = train_df[config.FEATURE_COLUMNS]
    y_train = train_df["bankrupt_label"]
    X_val = val_df[config.FEATURE_COLUMNS]
    y_val = val_df["bankrupt_label"]
    
    return X_train, y_train, X_val, y_val


def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200),
        "lgbm": LGBMClassifier()
    }
    best_model = None
    best_score = -1
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.joblib")
