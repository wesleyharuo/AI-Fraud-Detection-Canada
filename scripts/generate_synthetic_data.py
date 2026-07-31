"""Generate a synthetic transactions dataset with an explainable fraud rule.

The label is driven by a logistic combination of night-time activity,
cross-border transactions, transaction amount, channel, and customer
tenure -- so the trained model's risk factors line up with intuitive,
presentable reasons (useful for demos and explainability screenshots).
"""

import argparse
import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery",
    "ecommerce",
    "fuel",
    "travel",
    "electronics",
    "restaurant",
]
CHANNELS = ["pos", "web", "mobile", "atm"]
COUNTRIES = ["CA", "US", "MX", "GB", "FR", "IN"]
COUNTRY_P = [0.55, 0.12, 0.08, 0.08, 0.08, 0.09]
MERCHANT_P = [0.22, 0.22, 0.14, 0.10, 0.16, 0.16]
CHANNEL_P = [0.35, 0.28, 0.27, 0.10]
CATEGORY_AMOUNT_MULT = {
    "grocery": 0.6,
    "ecommerce": 1.1,
    "fuel": 0.5,
    "travel": 2.5,
    "electronics": 1.8,
    "restaurant": 0.5,
}


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    is_night = ((timestamps.hour >= 0) & (timestamps.hour <= 5)).astype(int)

    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n, p=MERCHANT_P)
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_P)
    country = rng.choice(COUNTRIES, size=n, p=COUNTRY_P)
    customer_age = rng.integers(18, 86, size=n)
    customer_tenure_months = rng.integers(0, 601, size=n)

    base_amount = rng.lognormal(mean=4.0, sigma=0.9, size=n)
    category_mult = pd.Series(merchant_category).map(CATEGORY_AMOUNT_MULT).values
    amount = np.clip(np.round(base_amount * category_mult, 2), 5, 9000)

    log_amount = np.log1p(amount)
    amount_z = (log_amount - log_amount.mean()) / log_amount.std()

    cross_border = (country != "CA").astype(int)
    new_customer = (customer_tenure_months < 6).astype(int)
    risky_channel = np.isin(channel, ["web", "mobile"]).astype(int)
    risky_merchant = np.isin(
        merchant_category, ["travel", "electronics", "ecommerce"]
    ).astype(int)
    noise = rng.normal(0, 1, size=n)

    logit = (
        -6.2
        + 1.9 * is_night
        + 1.6 * cross_border
        + 1.7 * amount_z
        + 1.0 * risky_channel
        + 1.1 * new_customer
        + 0.7 * risky_merchant
        + 0.6 * noise
    )
    prob = 1 / (1 + np.exp(-logit))
    label_fraud = (rng.uniform(0, 1, size=n) < prob).astype(int)

    return pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "timestamp": timestamps,
            "amount": amount,
            "merchant_category": merchant_category,
            "channel": channel,
            "country": country,
            "customer_age": customer_age,
            "customer_tenure_months": customer_tenure_months,
            "is_night": is_night,
            "label_fraud": label_fraud,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/transactions_sample.csv")
    args = p.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(
        f"Wrote {len(df)} rows to {args.out} (fraud rate={df['label_fraud'].mean():.3%})"
    )
