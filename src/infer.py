import argparse
import json
import pandas as pd
from fraudkit.model import load_model, predict_proba


def main(args):
    bundle = load_model("model/artifacts.joblib")
    row = {
        "amount": args.amount,
        "merchant_category": args.merchant_category,
        "channel": args.channel,
        "country": args.country,
        "customer_age": args.customer_age,
        "customer_tenure_months": args.customer_tenure_months,
        "is_night": args.is_night,
    }
    df = pd.DataFrame([row])
    # NOTE: the training pipeline includes encoders & one-hot inside the model
    proba = predict_proba(bundle, df)[0]
    result = {
        "fraud_probability": round(float(proba), 6),
        "threshold": bundle.threshold,
        "flagged": bool(proba >= bundle.threshold),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--amount", type=float, required=True)
    p.add_argument(
        "--merchant_category",
        type=str,
        required=True,
        choices=["grocery", "ecommerce", "fuel", "travel", "electronics", "restaurant"],
    )
    p.add_argument(
        "--channel", type=str, required=True, choices=["pos", "web", "mobile", "atm"]
    )
    p.add_argument(
        "--country",
        type=str,
        required=True,
        choices=["CA", "US", "MX", "GB", "FR", "IN"],
    )
    p.add_argument("--customer_age", type=int, required=True)
    p.add_argument("--customer_tenure_months", type=int, required=True)
    p.add_argument("--is_night", type=int, required=True, choices=[0, 1])
    main(p.parse_args())
