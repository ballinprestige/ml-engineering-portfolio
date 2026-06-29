"""FastAPI scoring service.

Loads the latest registered model (training one on first run if the registry is empty), validates
input against a Pydantic data contract, and returns a calibrated probability. Every prediction is
logged. Run locally with: `uvicorn service.app:app --reload` (with src/ on PYTHONPATH).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from . import registry, train
from .preprocessing import FEATURES
from .schemas import Features, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s service %(message)s")
log = logging.getLogger("service")
_state: dict = {}


def ensure_model():
    """Load the latest model; train and register one if the registry is empty."""
    if not registry.has_model():
        log.info("no model in registry — training a fresh one")
        train.train_and_register()
    model, meta = registry.load_latest()
    _state["model"], _state["meta"] = model, meta
    return model, meta


def _model():
    if "model" not in _state:
        ensure_model()
    return _state["model"], _state["meta"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _, meta = ensure_model()
    log.info("loaded model v%s (auc=%s)", meta["version"], meta["metrics"]["auc"])
    yield


app = FastAPI(
    title="ML Engineering Portfolio — scoring service", version="0.1.0", lifespan=lifespan
)


@app.get("/health")
def health():
    _, meta = _model()
    return {"status": "ok", "model_loaded": True, "model_version": meta["version"]}


@app.get("/model")
def model_info():
    _, meta = _model()
    return meta


@app.post("/predict", response_model=Prediction)
def predict(features: Features):
    model, meta = _model()
    x = np.array([[getattr(features, name) for name in FEATURES]], dtype=float)
    prob = float(model.predict_proba(x)[:, 1][0])
    pred = int(prob >= 0.5)
    log.info("predict prob=%.4f pred=%d model_v=%s", prob, pred, meta["version"])
    return Prediction(probability=round(prob, 4), prediction=pred, model_version=meta["version"])
