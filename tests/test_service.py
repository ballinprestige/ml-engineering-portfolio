"""Service tests via FastAPI's TestClient — liveness/readiness, lineage, the strict input
contract (out-of-range / missing / extra fields), and threshold/probability consistency.

A registered model is guaranteed by the session fixture in conftest.py (the release step).
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

import service.app as appmod
from service import registry
from service.app import app, load_model

VALID = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50,
}


def test_health_is_liveness():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_ready_and_model_lineage():
    with TestClient(app) as client:
        rd = client.get("/ready")
        assert rd.status_code == 200 and rd.json()["ready"] is True

        meta = client.get("/model").json()
        for key in (
            "git_sha",
            "training_seed",
            "data_sha256",
            "artifact_sha256",
            "hyperparameters",
            "split",
            "decision_threshold",
        ):
            assert key in meta, f"missing lineage field: {key}"
        assert "baseline_uncalibrated" in meta["metrics"]
        assert "calibrated" in meta["metrics"]


def test_predict_is_consistent_and_traceable():
    with TestClient(app) as client:
        r = client.post("/predict", json=VALID)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["probability"] <= 1.0
        # prediction is exactly (probability >= threshold) — they can never disagree
        assert body["prediction"] == int(body["probability"] >= body["threshold"])
        assert "X-Request-ID" in r.headers


def test_rejects_out_of_range_inputs():
    with TestClient(app) as client:
        absurd = {
            **VALID,
            "glucose": 999999,
            "blood_pressure": 99999,
            "age": 999,
            "pregnancies": 999,
        }
        assert client.post("/predict", json=absurd).status_code == 422


def test_rejects_missing_and_unexpected_fields():
    with TestClient(app) as client:
        assert client.post("/predict", json={"glucose": 100}).status_code == 422  # missing fields
        assert (
            client.post("/predict", json={**VALID, "ssn": "x"}).status_code == 422
        )  # extra forbidden


def test_startup_fails_without_artifact(tmp_path, monkeypatch):
    # the service refuses to serve when no model is registered (fail fast on an empty registry)
    monkeypatch.setattr(registry, "REGISTRY", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        load_model()


def test_ready_returns_503_when_no_model(monkeypatch):
    # /ready reports 503 until a model is loaded; build a client WITHOUT lifespan so state stays empty
    monkeypatch.setattr(appmod, "_state", {})
    client = TestClient(app)  # not used as a context manager -> lifespan does not run
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_prediction_emits_structured_log(caplog):
    with caplog.at_level(logging.INFO, logger="service"), TestClient(app) as client:
        client.post("/predict", json=VALID)
    parsed = [
        json.loads(rec.getMessage())
        for rec in caplog.records
        if rec.name == "service" and rec.getMessage().strip().startswith("{")
    ]
    prediction_logs = [p for p in parsed if p.get("event") == "prediction"]
    request_logs = [p for p in parsed if "latency_ms" in p]
    assert prediction_logs and "model_id" in prediction_logs[0]
    assert request_logs and "request_id" in request_logs[0]
