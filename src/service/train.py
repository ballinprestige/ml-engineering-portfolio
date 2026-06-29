"""Train the served model and register it with metrics + lineage.

The artifact is a single calibrated pipeline that takes RAW feature rows and returns calibrated
probabilities end to end: recode impossible zeros -> impute (train medians) -> scale -> gradient
boosting -> isotonic calibration. All preprocessing is fit on the training split only.
"""

from __future__ import annotations

import os

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


def train_and_register(seed: int = 42) -> dict:
    df = pd.read_csv(DATA)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["outcome"].to_numpy()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )

    pipe = Pipeline(
        [
            ("recode", FunctionTransformer(recode_zeros)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ]
    )
    pipe.fit(X_tr, y_tr)
    model = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(X_val, y_val)

    proba = model.predict_proba(X_te)[:, 1]
    metadata = {
        "model_type": "GradientBoosting + isotonic calibration",
        "metrics": {
            "auc": round(float(roc_auc_score(y_te, proba)), 4),
            "brier": round(float(brier_score_loss(y_te, proba)), 4),
        },
        "n_train": int(len(X_tr)),
        "features": FEATURES,
        "data_source": "data/pima_diabetes.csv (Pima Indians Diabetes, UCI/OpenML)",
        "sklearn_version": sklearn.__version__,
    }
    return registry.save_model(model, metadata)


if __name__ == "__main__":
    meta = train_and_register()
    print(f"registered model v{meta['version']}  metrics={meta['metrics']}")
