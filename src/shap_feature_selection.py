"""SHAP-driven feature selection.

High-dimensional models often carry redundant or noise features that add variance
without signal. Ranking features by mean absolute SHAP value and keeping only the
top-k preserves predictive performance while shrinking the model substantially.

`evaluate()` returns the metrics (importable + testable); `main()` adds the plots.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import shap
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
SEED = 42
DEFAULT_KS = (2, 4, 6, 8, 10, 15, 20, 30, 40)


def _fit_auc(X_tr, y_tr, X_te, y_te, cols, seed: int = SEED):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=4,
    )
    model.fit(X_tr[:, cols], y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te[:, cols])[:, 1])
    return auc, model


def evaluate(n: int = 12000, n_features: int = 40, ks=DEFAULT_KS, seed: int = SEED) -> dict[str, Any]:
    X, y = make_classification(
        n_samples=n, n_features=n_features, n_informative=8, n_redundant=6,
        n_repeated=0, random_state=seed,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

    full_auc, model = _fit_auc(X_tr, y_tr, X_te, y_te, np.arange(n_features), seed)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1]

    ks = [k for k in ks if k <= n_features]
    aucs = [float(_fit_auc(X_tr, y_tr, X_te, y_te, order[:k], seed)[0]) for k in ks]
    top10 = float(_fit_auc(X_tr, y_tr, X_te, y_te, order[:10], seed)[0]) if n_features >= 10 else aucs[-1]
    return {
        "ks": list(ks),
        "aucs": aucs,
        "full_auc": float(full_auc),
        "top10_auc": top10,
        "n_features": n_features,
        "shap_values": sv,
        "X_te": X_te,
    }


def main() -> None:
    r = evaluate()
    for k, a in zip(r["ks"], r["aucs"]):
        print(f"top-{k:2d} features  AUC {a:.3f}")
    print(f"all {r['n_features']} features  AUC {r['full_auc']:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(r["ks"], r["aucs"], "o-", color="#1f3a5f")
    ax.axhline(r["full_auc"], ls="--", color="#888", label=f"all {r['n_features']} features ({r['full_auc']:.3f})")
    ax.set_xlabel("number of top-SHAP features kept")
    ax.set_ylabel("test ROC AUC")
    ax.set_title("SHAP-ranked feature selection preserves performance")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "shap_feature_selection.png"), dpi=130)

    plt.figure()
    shap.summary_plot(r["shap_values"], r["X_te"], plot_type="bar", max_display=12, show=False)
    plt.title("Mean |SHAP| feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, "shap_importance.png"), dpi=130, bbox_inches="tight")
    print("saved images")


if __name__ == "__main__":
    main()
