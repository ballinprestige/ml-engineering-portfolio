"""Registry integrity tests — atomic publication and checksum-verified loads."""

import json
import os

import pytest

from service import registry, train


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY", str(tmp_path / "registry"))
    meta = train.train_and_register(seed=0)
    assert registry.has_model()
    model, loaded = registry.load_latest()
    assert loaded["version"] == meta["version"]
    assert loaded["artifact_sha256"] == meta["artifact_sha256"]
    assert hasattr(model, "predict_proba")


def test_load_detects_corrupted_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY", str(tmp_path / "registry"))
    train.train_and_register(seed=0)
    version = json.load(open(os.path.join(registry.REGISTRY, "latest.json")))["version"]
    artifact = os.path.join(registry.REGISTRY, f"v{version}", "model.joblib")
    with open(artifact, "ab") as f:
        f.write(b"corruption")  # tamper the bytes -> checksum no longer matches
    with pytest.raises(ValueError):
        registry.load_latest()
