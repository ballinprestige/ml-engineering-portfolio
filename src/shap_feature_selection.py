"""SHAP-driven feature selection.

High-dimensional models often carry many redundant or noise features that add
variance without signal. This demo trains an XGBoost classifier on a dataset
with a few informative features and many noise features, ranks features by mean
absolute SHAP value, and shows that pruning to the top-k features preserves
predictive performance while shrinking the model substantially.
"""
import os

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
N_FEATURES = 40


def fit_auc(X_tr, y_tr, X_te, y_te, cols):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=4,
    )
    model.fit(X_tr[:, cols], y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te[:, cols])[:, 1])
    return auc, model


def main():
    X, y = make_classification(
        n_samples=12000, n_features=N_FEATURES, n_informative=8, n_redundant=6,
        n_repeated=0, random_state=SEED,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    full_auc, model = fit_auc(X_tr, y_tr, X_te, y_te, np.arange(N_FEATURES))

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1]

    ks = [2, 4, 6, 8, 10, 15, 20, 30, N_FEATURES]
    aucs = []
    for k in ks:
        auc, _ = fit_auc(X_tr, y_tr, X_te, y_te, order[:k])
        aucs.append(auc)
        print(f"top-{k:2d} features  AUC {auc:.3f}")
    print(f"all {N_FEATURES} features  AUC {full_auc:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, aucs, "o-", color="#1f3a5f")
    ax.axhline(full_auc, ls="--", color="#888", label=f"all {N_FEATURES} features ({full_auc:.3f})")
    ax.set_xlabel("number of top-SHAP features kept")
    ax.set_ylabel("test ROC AUC")
    ax.set_title("SHAP-ranked feature selection preserves performance")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "shap_feature_selection.png"), dpi=130)

    plt.figure()
    shap.summary_plot(sv, X_te, plot_type="bar", max_display=12, show=False)
    plt.title("Mean |SHAP| feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES, "shap_importance.png"), dpi=130, bbox_inches="tight")
    print("saved images")


if __name__ == "__main__":
    main()
