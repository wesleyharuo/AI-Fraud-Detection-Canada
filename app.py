from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
from fraudkit.model import load_model, predict_proba
from fraudkit.schema import Transaction

# Prometheus
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = FastAPI(title="Canada Fraud Detection API", version="1.0")
bundle = load_model("model/artifacts.joblib")

REQ_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"])
REQ_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    endpoint = request.url.path
    REQ_COUNT.labels(endpoint=endpoint, method=request.method, status=str(response.status_code)).inc()
    REQ_LATENCY.labels(endpoint=endpoint).observe(latency)
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return FastAPI.responses.Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/score")
def score(tx: Transaction):
    df = pd.DataFrame([tx.model_dump()])
    proba = float(predict_proba(bundle, df)[0])
    flagged = proba >= bundle.threshold
    return {
        "fraud_probability": round(proba, 6),
        "threshold": bundle.threshold,
        "flagged": flagged,
        "explanations": [
            "Night transactions and high amounts tend to increase risk.",
            "Cross-border transactions may increase probability."
        ]
    }
