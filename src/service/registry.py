"""A minimal model registry with versioning and artifact lineage.

Each registered model gets its own versioned directory holding the serialized artifact and a
metadata.json (metrics, training rows, feature list, data source, library versions, timestamp).
A `latest.json` pointer records the current version. This is the small, honest core of what a
real registry (MLflow, etc.) does.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY = os.path.join(ROOT, "models", "registry")


def _next_version() -> int:
    if not os.path.isdir(REGISTRY):
        return 1
    versions = [int(d[1:]) for d in os.listdir(REGISTRY) if d.startswith("v") and d[1:].isdigit()]
    return (max(versions) + 1) if versions else 1


def save_model(model, metadata: dict[str, Any]) -> dict[str, Any]:
    os.makedirs(REGISTRY, exist_ok=True)
    version = _next_version()
    vdir = os.path.join(REGISTRY, f"v{version}")
    os.makedirs(vdir, exist_ok=True)
    joblib.dump(model, os.path.join(vdir, "model.joblib"))
    meta = {
        **metadata,
        "version": version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(vdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(REGISTRY, "latest.json"), "w") as f:
        json.dump({"version": version}, f)
    return meta


def has_model() -> bool:
    return os.path.exists(os.path.join(REGISTRY, "latest.json"))


def load_latest():
    latest = os.path.join(REGISTRY, "latest.json")
    if not os.path.exists(latest):
        raise FileNotFoundError("no model registered — run service.train.train_and_register()")
    with open(latest) as f:
        version = json.load(f)["version"]
    vdir = os.path.join(REGISTRY, f"v{version}")
    model = joblib.load(os.path.join(vdir, "model.joblib"))
    with open(os.path.join(vdir, "metadata.json")) as f:
        meta = json.load(f)
    return model, meta
