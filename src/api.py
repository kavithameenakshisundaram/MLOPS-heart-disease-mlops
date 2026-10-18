import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "heart_disease_model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger("heart-disease-api")

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"]
)

model = joblib.load(MODEL_PATH)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        REQUEST_COUNT.labels(
            request.method,
            request.url.path,
            str(locals().get("status", 500))
        ).inc()
        REQUEST_LATENCY.labels(request.url.path).observe(duration)

        logger.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            locals().get("status", 500),
            duration * 1000
        )


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(payload: dict[str, Any]):
    try:
        input_df = pd.DataFrame([payload])

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities))

        return {
            "prediction": prediction,
            "prediction_label": (
                "heart_disease_present"
                if prediction == 1
                else "heart_disease_absent"
            ),
            "confidence": round(confidence, 4)
        }

    except Exception as exc:
        logger.exception("Prediction failed")
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)}
        )


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
