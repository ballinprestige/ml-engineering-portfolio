"""FastAPI scoring service.

Loads one immutable, checksum-verified model artifact at startup and **fails fast** if none is
available — training is a separate release step (`python -m service.train`), not something the
service does on the fly. Validates input against a strict Pydantic contract, returns a calibrated
probability with the model's documented decision threshold, and emits one structured JSON log line
per request (with a request id and latency).

Run: `python -m service.train && uvicorn service.app:app`
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request, Response, status

from . import registry
from .preprocessing import FEATURES
from .schemas import Features, Prediction

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("service")
_state: dict = {}


def load_model():
    """Load the latest registered artifact; fail fast (raise) if absent or schema-incompatible."""
    model, meta = registry.load_latest()
    if meta.get("features") != FEATURES:
        raise RuntimeError("model/feature-schema mismatch — refusing to serve")
    _state["model"], _state["meta"] = model, meta
    return model, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, meta = load_model()  # raises -> app fails to start (fail fast)
    log.info(
        json.dumps(
            {"event": "startup", "model_version": meta["version"], "metrics": meta["metrics"]}
        )
    )
    yield


app = FastAPI(
    title="ML Engineering Portfolio — scoring service", version="1.0.0", lifespan=lifespan
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log.info(
        json.dumps(
            {
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )
    )
    return response


@app.get("/health")
def health():
    """Liveness — the process is up (independent of model state)."""
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response):
    """Readiness — a model is loaded and the service can serve traffic."""
    if "model" in _state:
        return {"ready": True, "model_version": _state["meta"]["version"]}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": False}


@app.get("/model")
def model_info():
    return _state["meta"]


@app.post("/predict", response_model=Prediction)
def predict(features: Features, request: Request):
    meta = _state["meta"]
    threshold = meta["decision_threshold"]
    x = np.array([[getattr(features, name) for name in FEATURES]], dtype=float)
    raw = float(_state["model"].predict_proba(x)[:, 1][0])
    pred = int(raw >= threshold)  # decide on the RAW probability, never the rounded display value
    probability = round(raw, 6)
    log.info(
        json.dumps(
            {
                "request_id": request.state.request_id,
                "event": "prediction",
                "probability": probability,
                "prediction": pred,
                "model_id": meta.get("model_id"),
                "model_version": meta["version"],
            }
        )
    )
    return Prediction(
        probability=probability,
        prediction=pred,
        threshold=threshold,
        model_version=meta["version"],
        model_id=meta.get("model_id", "unknown"),
    )
