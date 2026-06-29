"""Service tests via FastAPI's TestClient (no running server / Docker needed).

Using TestClient as a context manager triggers the app lifespan, which trains and registers a
model if the registry is empty — so these tests exercise the full train -> register -> serve path.
"""

from fastapi.testclient import TestClient

from service.app import app


def test_health_and_model_metadata():
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"
        assert h.json()["model_version"] >= 1

        m = client.get("/model")
        assert m.status_code == 200
        assert "auc" in m.json()["metrics"]
        assert m.json()["features"][1] == "glucose"


def test_predict_returns_calibrated_probability():
    with TestClient(app) as client:
        payload = {
            "pregnancies": 6,
            "glucose": 148,
            "blood_pressure": 72,
            "skin_thickness": 35,
            "insulin": 0,
            "bmi": 33.6,
            "diabetes_pedigree": 0.627,
            "age": 50,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["prediction"] in (0, 1)


def test_predict_rejects_invalid_input():
    with TestClient(app) as client:
        r = client.post("/predict", json={"glucose": 100})  # missing required fields
        assert r.status_code == 422
