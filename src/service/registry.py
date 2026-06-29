"""A small, honest **single-process local artifact store**.

It is intentionally minimal, not a distributed registry. What it does guarantee:
- **Atomic publication**: artifacts and metadata are written to a temp file and `os.replace`d,
  and `latest.json` is published *last*, so a reader never sees a half-written version.
- **Integrity**: every artifact records a SHA-256, verified on load (corruption fails loudly).

What it does NOT do (by design — training is a single-writer release step, not concurrent):
multi-writer locking, durable/remote storage, or stage management. For real deployments back
this with object storage or MLflow. Naming it an "artifact store" rather than a "registry"
keeps the claim honest.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
from typing import Any

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY = os.path.join(ROOT, "models", "registry")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _write_json_atomic(path: str, obj: dict) -> None:
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


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

    model_path = os.path.join(vdir, "model.joblib")
    tmp = model_path + ".tmp"
    joblib.dump(model, tmp)
    os.replace(tmp, model_path)

    meta = {
        **metadata,
        "version": version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifact_sha256": _sha256(model_path),
    }
    _write_json_atomic(os.path.join(vdir, "metadata.json"), meta)
    # publish the pointer LAST, atomically — readers never see a partial version
    _write_json_atomic(os.path.join(REGISTRY, "latest.json"), {"version": version})
    return meta


def has_model() -> bool:
    return os.path.exists(os.path.join(REGISTRY, "latest.json"))


def load_latest():
    latest = os.path.join(REGISTRY, "latest.json")
    if not os.path.exists(latest):
        raise FileNotFoundError(
            "no model registered — training is a release step: run `python -m service.train`"
        )
    with open(latest) as f:
        version = json.load(f)["version"]
    vdir = os.path.join(REGISTRY, f"v{version}")
    with open(os.path.join(vdir, "metadata.json")) as f:
        meta = json.load(f)

    model_path = os.path.join(vdir, "model.joblib")
    actual = _sha256(model_path)
    if actual != meta.get("artifact_sha256"):
        raise ValueError(
            f"artifact checksum mismatch for v{version} — registry corrupt; refusing to load"
        )
    return joblib.load(model_path), meta
