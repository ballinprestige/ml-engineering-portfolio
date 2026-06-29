"""SHAP-driven feature selection (leakage-safe protocol).

The honest protocol matters here: feature ranking and the choice of *how many* features to
keep are decisions, and decisions made on the test set leak information. So:

* train split  -> fit the model
* validation split -> rank features by mean |SHAP| and CHOOSE k (smallest k within tolerance
  of the best validation AUC)
* untouched test split -> report ONE final number

We repeat the whole protocol across seeds and report mean +/- std, so the result is a genuine
out-of-sample estimate rather than a single lucky point.
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
SEEDS = range(10)
GRID = (2, 4, 6, 8, 10, 15, 20, 30, 40)
TOL = 0.005


def _fit_auc(X_tr, y_tr, X_eval, y_eval, cols, seed):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr[:, cols], y_tr)
    auc = roc_auc_score(y_eval, model.predict_proba(X_eval[:, cols])[:, 1])
    return auc, model


def evaluate(
    n: int = 12000, n_features: int = 40, seeds=SEEDS, grid=GRID, tol: float = TOL
) -> dict[str, Any]:
    grid = tuple(k for k in grid if k <= n_features)
    full_test, topk_test, chosen_k = [], [], []
    rep = None
    for si, seed in enumerate(seeds):
        X, y = make_classification(
            n_samples=n,
            n_features=n_features,
            n_informative=8,
            n_redundant=6,
            n_repeated=0,
            random_state=seed,
        )
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.5, random_state=seed, stratify=y
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
        )

        all_cols = np.arange(n_features)
        full_auc, model = _fit_auc(X_tr, y_tr, X_te, y_te, all_cols, seed)  # report on TEST
        full_test.append(full_auc)

        sv = shap.TreeExplainer(model).shap_values(X_val)  # rank on VALIDATION
        order = np.argsort(np.abs(sv).mean(axis=0))[::-1]

        val_aucs = {k: _fit_auc(X_tr, y_tr, X_val, y_val, order[:k], seed)[0] for k in grid}
        best_val = max(val_aucs.values())
        k_star = min(k for k in grid if val_aucs[k] >= best_val - tol)  # CHOOSE k on VALIDATION

        topk_auc = _fit_auc(X_tr, y_tr, X_te, y_te, order[:k_star], seed)[0]  # report on TEST
        topk_test.append(topk_auc)
        chosen_k.append(k_star)
        if si == 0:
            rep = {"order": order, "sv": sv, "X_val": X_val, "val_aucs": val_aucs, "k_star": k_star}

    full_test, topk_test = np.array(full_test), np.array(topk_test)
    return {
        "full_test_mean": float(full_test.mean()),
        "full_test_std": float(full_test.std()),
        "topk_test_mean": float(topk_test.mean()),
        "topk_test_std": float(topk_test.std()),
        "chosen_k": [int(k) for k in chosen_k],
        "k_median": int(np.median(chosen_k)),
        "n_features": n_features,
        "n_seeds": len(list(seeds)),
        "rep": rep,
    }


def main() -> None:
    r = evaluate()
    print(
        f"full {r['n_features']} features  test AUC: {r['full_test_mean']:.3f} +/- {r['full_test_std']:.3f}"
    )
    print(
        f"validation-selected k (median {r['k_median']})  test AUC: {r['topk_test_mean']:.3f} +/- {r['topk_test_std']:.3f}"
    )
    print(f"chosen k per seed: {r['chosen_k']}")

    rep = r["rep"]
    ks = sorted(rep["val_aucs"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, [rep["val_aucs"][k] for k in ks], "o-", color="#1f3a5f", label="validation AUC")
    ax.axvline(
        rep["k_star"], ls="--", color="#c0392b", label=f"k chosen on validation = {rep['k_star']}"
    )
    ax.set_xlabel("number of top-SHAP features kept")
    ax.set_ylabel("validation ROC AUC")
    ax.set_title(
        f"Feature count is chosen on validation; reported once on test\n"
        f"(test: all {r['n_features']} = {r['full_test_mean']:.3f}±{r['full_test_std']:.3f}, "
        f"selected = {r['topk_test_mean']:.3f}±{r['topk_test_std']:.3f})",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "shap_feature_selection.png"), dpi=130)

    plt.figure()
    shap.summary_plot(rep["sv"], rep["X_val"], plot_type="bar", max_display=12, show=False)
    plt.title("Mean |SHAP| feature importance (validation split)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, "shap_importance.png"), dpi=130, bbox_inches="tight")
    print("saved images")


if __name__ == "__main__":
    main()
