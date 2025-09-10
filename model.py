import joblib, json, os
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np

@dataclass
class ModelBundle:
    model: Any
    encoders: Dict
    feature_cols: list
    threshold: float

def save_model(bundle: ModelBundle, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({
        "model": bundle.model,
        "encoders": bundle.encoders,
        "feature_cols": bundle.feature_cols,
        "threshold": bundle.threshold
    }, path)

def load_model(path: str) -> ModelBundle:
    blob = joblib.load(path)
    return ModelBundle(**blob)

def predict_proba(bundle: ModelBundle, X) -> np.ndarray:
    return bundle.model.predict_proba(X)[:,1]
