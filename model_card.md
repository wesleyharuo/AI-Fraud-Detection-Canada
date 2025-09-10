# 🧾 Model Card: Canada Fraud Detection (CatBoost)

## Model Overview
- **Algorithm:** CatBoostClassifier
- **Task:** Binary classification (fraud vs. legitimate transactions)
- **Version:** v1.1 (CatBoost swap)
- **Author:** Wesley Kurosawa + ChatGPT

## Intended Use
- Fraud detection across Canadian financial institutions (banks, credit unions, PSPs).
- To be deployed via FastAPI microservice or Docker container.
- Outputs fraud probability + binary flag.

## Limitations
- Training data here is **synthetic**. Real-world deployment requires retraining on actual historical data.
- Does not include all possible fraud modalities (e.g., synthetic identities, collusion).

## Metrics (synthetic baseline)
- ROC-AUC: ~0.90
- PR-AUC: ~0.45
- F1: ~0.60 (with threshold 0.5)

## Fairness & Compliance
- **Canadian PIPEDA compliance:** PII minimization, explainability, human-in-loop.
- **Bias risk:** must test with real data to ensure no discrimination (e.g., age, postal codes).

## Explainability
- SHAP integration available.
- Each prediction can be explained for adverse action documentation.

## Monitoring
- Evidently integrated for drift detection.
- Retrain cadence: every 3-6 months or after drift.
