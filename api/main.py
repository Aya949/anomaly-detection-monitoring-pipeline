from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from schemas import (
    BatchInput,
    BatchPredictionResponse,
    HealthResponse,
)
from predictor import predictor
from alerting import send_slack_alert

# ── Custom Prometheus metrics ─────────────────────────────────────────────────
anomaly_counter = Counter(
    "anomalies_detected_total",
    "Total number of anomalies detected"
)
records_counter = Counter(
    "records_scored_total",
    "Total number of records scored"
)
anomaly_rate_gauge = Gauge(
    "anomaly_rate_percent",
    "Current anomaly rate percentage"
)
inference_latency = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Anomaly Detection API is ready")
    yield
    print("[shutdown] Cleaning up")


app = FastAPI(
    title       = "Anomaly Detection API",
    description = "Real-time server metrics anomaly detection "
                  "using Isolation Forest trained on NAB AWS CloudWatch data.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── Prometheus instrumentator — must be AFTER app is created ─────────────────
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return {
        "status"        : "ok",
        "model_loaded"  : predictor.model is not None,
        "feature_count" : len(predictor.feature_cols),
        "threshold"     : predictor.config["best_threshold"],
        "roc_auc"       : predictor.config["roc_auc"],
    }


@app.post("/predict", response_model=BatchPredictionResponse,
          tags=["Prediction"])
def predict(payload: BatchInput):
    """
    Run anomaly detection on a batch of server metrics.
    Minimum 37 rows required for rolling window features.
    Automatically sends Slack alert if anomalies are detected.
    """
    try:
        start   = time.time()
        records = [r.model_dump() for r in payload.records]
        result  = predictor.predict(records)
        elapsed = round(time.time() - start, 4)

        # ── Update Prometheus metrics ─────────────────────────────────────
        records_counter.inc(result["total_records"])
        anomaly_counter.inc(result["anomaly_count"])
        anomaly_rate_gauge.set(result["anomaly_rate_pct"])
        inference_latency.observe(elapsed)

        print(f"[predict] {result['total_records']} records | "
              f"{result['anomaly_count']} anomalies | {elapsed}s")

        # ── Send Slack alert if anomalies found ───────────────────────────
        if result["anomaly_count"] > 0:
            anomalies = [p for p in result["predictions"]
                         if p["is_anomaly"] == 1]
            send_slack_alert(anomalies, result)

        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["System"])
def root():
    return {
        "message" : "Anomaly Detection API",
        "docs"    : "/docs",
        "health"  : "/health",
    }