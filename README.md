# 🇨🇦 AI Fraud Detection for Financial Systems (Canada-ready)

**Production-grade, open-source template** for detecting transactional fraud across Canadian financial systems (works generically for banks, credit unions, and payment processors). 
Built with a modern ML stack: `pandas/polars`, `scikit-learn`, `CatBoost`, `FastAPI`, `SHAP`, and `Evidently` for drift/monitoring.

##  Highlights
- Robust feature pipeline (time & category encoding, train/test leakage guards)
- Strong baseline model with CatBoost (easily swap to CatBoost/XGBoost)
- Model cards + explainability via SHAP
- FastAPI **real-time scoring API** with request/response schema validation
- Monitoring hooks using **Evidently** (data & performance drift)
- Privacy‑aware design (PII minimization & audit log stubs)
- Sample dataset & unit tests so you can run in minutes

##  Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Train & evaluate
python src/train.py --config config.yaml

# 2) Local inference
python src/infer.py --config config.yaml --amount 350 --merchant_category grocery --channel pos --country CA --customer_age 35 --customer_tenure_months 24 --is_night 0

# 3) Run API
uvicorn api.app:app --reload --port 8000
# POST http://localhost:8000/score
```

##  Structure
```
ai_fraud_detection_canada/
├── api/
│   └── app.py
├── data/
│   └── transactions_sample.csv
├── src/
│   ├── fraudkit/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── features.py
│   │   ├── model.py
│   │   ├── explain.py
│   │   ├── monitoring.py
│   │   └── utils.py
│   ├── train.py
│   └── infer.py
├── tests/
│   └── test_pipeline.py
├── config.yaml
└── requirements.txt
```

##  Canadian context (PIPEDA & banking)
- **PII minimization:** only non-identifying features used in the sample; add tokenization/Hashing for IDs if needed.
- **Explainability:** SHAP summary + per‑prediction reasons for adverse action documentation.
- **Human‑in‑the‑loop:** threshold controls & review flags before blocking/declining transactions.

##  Reproducible baseline scores (synthetic data)
- ROC‑AUC ~ 0.90 ±0.02, PR‑AUC ~ 0.45 ±0.05 (varies; synthetic)
- Use your bank’s historical data for a true benchmark.

##  License
MIT. Attribution appreciated: “Template by Wesley Kurosawa + ChatGPT.”

---

##  Docker
Build and run the API in Docker:
```bash
docker build -t ai-fraud-canada:latest .
docker run -p 8000:8000 ai-fraud-canada:latest
```

##  CI (GitHub Actions)
Workflow runs on push/PR:
- Ruff lint
- Black format check
- Pytest
- Docker build (+ push if `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets are set and branch is `main` or a tag `v*`).

##  Documentation
- See `MODEL_CARD.md` for model details and `adverse_action_template.md` for compliant customer notices.

##  Monitoring (Compose)
Use Docker Compose to run API + Prometheus + Grafana stack:
```bash
docker-compose up --build
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

##  CI Artifacts & Releases
- GitHub Actions now uploads `model/eval_sample.csv` and SHAP plots (if generated) as artifacts.
- On tagged releases (`v*`), automated release notes are drafted via Release Drafter.

##  Metrics, Prometheus & Grafana (docker-compose)
Start the full stack (API + Prometheus + Grafana):
```bash
docker compose up -d
# API:        http://localhost:8000
# Metrics:    http://localhost:8000/metrics
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
```
Prometheus is pre-configured to scrape the API at `/metrics`. Grafana comes empty—you can add Prometheus as a datasource (`http://prometheus:9090`) and import a FastAPI/Prometheus dashboard.
