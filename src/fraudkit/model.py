import joblib
import os
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    model: Any
    encoders: Dict
    feature_cols: list
    threshold: float


def save_model(bundle: ModelBundle, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "encoders": bundle.encoders,
            "feature_cols": bundle.feature_cols,
            "threshold": bundle.threshold,
        },
        path,
    )


def load_model(path: str) -> ModelBundle:
    blob = joblib.load(path)
    return ModelBundle(**blob)


def prepare_features(bundle: ModelBundle, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature pipeline used at training time (frequency
    encoders + column order) so raw request rows match what the model
    was fit on."""
    from fraudkit.features import build_features, transform_with_encoders, CATEGORICAL

    X = build_features(X)
    X = transform_with_encoders(X, bundle.encoders)
    for c in CATEGORICAL:
        X[c] = X[c].astype("string")
    return X[bundle.feature_cols]


def predict_proba(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    X = prepare_features(bundle, X)
    return bundle.model.predict_proba(X)[:, 1]
