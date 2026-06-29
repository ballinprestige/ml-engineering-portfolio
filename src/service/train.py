"""Train the served model and register it — the **release step**, separate from serving.

Produces one immutable, checksummed artifact (a calibrated end-to-end pipeline: recode impossible
zeros -> impute on train medians -> scale -> gradient boosting -> sigmoid/Platt calibration —
sigmoid, not isotonic, because the calibration split is small and isotonic's ties hurt ranking). The
metadata captures enough lineage to reproduce the run: data + artifact checksums, git commit,
seed, split definition, hyperparameters, library/python versions, decision threshold, and both
baseline (uncalibrated) and calibrated metrics.

Run as: `python -m service.train`
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess

import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import registry
from .preprocessing import FEATURES, recode_zeros

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "pima_diabetes.csv")
DECISION_THRESHOLD = 0.5  # default; should be tuned to the cost of false +/- (see model_card.md)


def _git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _metrics(y_true, proba) -> dict:
    return {
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


def train_and_register(seed: int = 42) -> dict:
    df = pd.read_csv(DATA)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["outcome"].to_numpy()

    # 70 / 15 / 15 train / calibration / test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    clf = GradientBoostingClassifier(random_state=seed)
    pipe = Pipeline(
        [
            ("recode", FunctionTransformer(recode_zeros)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )
    pipe.fit(X_tr, y_tr)
    baseline = _metrics(y_te, pipe.predict_proba(X_te)[:, 1])

    # sigmoid (Platt), not isotonic: the calibration split is small, and isotonic's step ties
    # would degrade ranking/AUC. Sigmoid is smooth and preserves the ranking.
    model = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(X_cal, y_cal)
    calibrated = _metrics(y_te, model.predict_proba(X_te)[:, 1])

    hp = clf.get_params()
    metadata = {
        "model_type": "GradientBoostingClassifier + sigmoid (Platt) calibration",
        "framework": f"scikit-learn {sklearn.__version__}",
        "python_version": platform.python_version(),
        "git_sha": _git_sha(),
        "training_seed": seed,
        "data_source": "Pima Indians Diabetes (UCI/OpenML id 37)",
        "data_file": "data/pima_diabetes.csv",
        "data_sha256": _sha256_file(DATA),
        "split": {"train": int(len(X_tr)), "calibration": int(len(X_cal)), "test": int(len(X_te))},
        "hyperparameters": {
            "n_estimators": hp["n_estimators"],
            "learning_rate": hp["learning_rate"],
            "max_depth": hp["max_depth"],
            "subsample": hp["subsample"],
        },
        "calibration": "sigmoid",
        "decision_threshold": DECISION_THRESHOLD,
        "features": FEATURES,
        "metrics": {"baseline_uncalibrated": baseline, "calibrated": calibrated},
        "metrics_note": (
            "single seeded train/cal/test split; the README demo's 0.81±0.03 AUC is a "
            "cross-split variability estimate, not this artifact's number"
        ),
    }
    return registry.save_model(model, metadata)


if __name__ == "__main__":
    meta = train_and_register()
    print(
        f"registered model v{meta['version']}  metrics={meta['metrics']}  sha={meta['artifact_sha256'][:12]}"
    )
