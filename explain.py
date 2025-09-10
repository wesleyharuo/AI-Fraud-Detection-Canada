import shap
import numpy as np

def shap_explainer(model, X_sample):
    # TreeExplainer for LightGBM; fall back to KernelExplainer if needed
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        return explainer, shap_vals
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
        shap_vals = explainer.shap_values(X_sample[:200])
        return explainer, shap_vals
