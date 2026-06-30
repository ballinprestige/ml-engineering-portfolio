"""Registry integrity tests — atomic publication and checksum-verified loads."""

import json
import os
import shutil

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


def test_interrupted_publication_is_invisible(tmp_path, monkeypatch):
    # a half-published version (artifact on disk, latest.json never updated) must not be served:
    # readers only ever see the atomically published pointer.
    monkeypatch.setattr(registry, "REGISTRY", str(tmp_path / "registry"))
    train.train_and_register(seed=0)  # publishes v1 -> latest.json points at v1
    v1 = os.path.join(registry.REGISTRY, "v1")
    v2 = os.path.join(registry.REGISTRY, "v2")
    shutil.copytree(v1, v2)  # simulate an interrupted v2 write with no latest.json bump
    _, loaded = registry.load_latest()
    assert loaded["version"] == 1
