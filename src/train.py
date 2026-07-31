import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from fraudkit.features import (
    build_features,
    fit_encoders,
    transform_with_encoders,
    CATEGORICAL,
    NUMERICAL,
)
from fraudkit.model import ModelBundle, save_model
from fraudkit.utils import load_config


def main(config_path: str):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    target = data_cfg["target"]
    dt_col = data_cfg.get("datetime_col")
    threshold = cfg["compliance"].get("threshold", 0.5)

    df = pd.read_csv(data_cfg["train_csv"], parse_dates=[dt_col] if dt_col else None)
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    # Base features
    X = build_features(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    # Fit leakage-safe encoders and add frequency features
    encoders = fit_encoders(X_train)
    X_train = transform_with_encoders(X_train, encoders)
    X_test = transform_with_encoders(X_test, encoders)

    # Prepare feature columns: CatBoost can ingest raw categoricals + numeric/freq
    freq_cols = [f"{c}_freq" for c in CATEGORICAL]
    feature_cols = NUMERICAL + freq_cols + CATEGORICAL

    # Ensure dtype for categorical columns is string/object
    for c in CATEGORICAL:
        X_train[c] = X_train[c].astype("string")
        X_test[c] = X_test[c].astype("string")

    # Map cat feature indices relative to feature_cols
    cat_feature_indices = [feature_cols.index(c) for c in CATEGORICAL]

    # Train CatBoost
    from catboost import CatBoostClassifier, Pool

    params = model_cfg.get("params", {})
    clf = CatBoostClassifier(**params, verbose=False, eval_metric="AUC")
    train_pool = Pool(
        X_train[feature_cols], label=y_train, cat_features=cat_feature_indices
    )
    test_pool = Pool(
        X_test[feature_cols], label=y_test, cat_features=cat_feature_indices
    )
    clf.fit(train_pool, eval_set=test_pool)

    # Evaluate
    proba = clf.predict_proba(test_pool)[:, 1]
    pred = (proba >= threshold).astype(int)

    roc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    print(json.dumps({"roc_auc": roc, "average_precision": ap, "f1": f1}, indent=2))

    # Save bundle
    bundle = ModelBundle(
        model=clf, encoders=encoders, feature_cols=feature_cols, threshold=threshold
    )
    os.makedirs("model", exist_ok=True)
    save_model(bundle, "model/artifacts.joblib")

    # Save eval for monitoring demo
    eval_df = X_test[feature_cols].copy()
    eval_df["proba"] = proba
    eval_df["pred"] = pred
    eval_df[target] = y_test.values
    eval_df.to_csv("model/eval_sample.csv", index=False)

    # Generate SHAP plots (using CatBoost-specific SHAP values)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use a small sample for speed
        sample_idx = np.random.default_rng(cfg["training"]["random_state"]).choice(
            len(X_test), size=min(500, len(X_test)), replace=False
        )
        Xs = X_test.iloc[sample_idx][feature_cols]
        ys = y_test.iloc[sample_idx]
        shap_pool = Pool(Xs, label=ys, cat_features=cat_feature_indices)
        shap_vals = clf.get_feature_importance(shap_pool, type="ShapValues")
        # shap_vals shape: (n_samples, n_features + 1) last column is expected value
        contribs = shap_vals[:, :-1]
        mean_abs = np.abs(contribs).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        names = np.array(feature_cols)[order]

        # Bar plot of mean |SHAP|
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(names)), mean_abs[order])
        plt.xticks(range(len(names)), names, rotation=60, ha="right")
        plt.title("Mean |SHAP| Feature Importance")
        plt.tight_layout()
        plt.savefig("model/shap_importance.png", dpi=150)
        plt.close()

        # Summary-like scatter for top 15
        topk = min(15, len(names))
        plt.figure(figsize=(10, 6))
        for i in range(topk):
            plt.scatter([i] * len(sample_idx), contribs[:, order[i]], s=4, alpha=0.3)
        plt.xticks(range(topk), names[:topk], rotation=60, ha="right")
        plt.title("SHAP Contributions (Top Features)")
        plt.tight_layout()
        plt.savefig("model/shap_summary.png", dpi=150)
        plt.close()
    except Exception as e:
        print("SHAP plotting skipped:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
