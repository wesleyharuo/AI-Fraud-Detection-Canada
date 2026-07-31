import pandas as pd

CATEGORICAL = ["merchant_category", "channel", "country"]
NUMERICAL = ["amount", "customer_age", "customer_tenure_months", "is_night"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Amount sanity
    df["amount"] = df["amount"].clip(lower=0)

    # Frequency encodings (leak‑safe: fit only on train in train.py)
    return df


def fit_encoders(train_df: pd.DataFrame):
    encoders = {}
    for col in CATEGORICAL:
        freqs = train_df[col].value_counts(normalize=True).to_dict()
        encoders[col] = freqs
    return encoders


def transform_with_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in encoders.items():
        df[f"{col}_freq"] = df[col].map(mapping).fillna(0.0)
    return df
