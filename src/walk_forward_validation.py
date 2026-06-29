"""Leakage-safe time-series validation.

Random K-fold cross-validation quietly leaks future information on time-ordered
data: shuffling places samples from every period into training, so the model is
scored on conditions it has already seen. Walk-forward validation always trains
on the past and tests on the future, matching how a model is actually deployed.

`evaluate()` returns the metrics (importable + testable); `main()` adds the plot.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
SEED = 42


def make_drifting_timeseries(n: int = 4000, n_features: int = 10, seed: int = SEED):
    """Time-ordered data with a regime switch (concept drift).

    Early in the series the signal lives in feature 1; later it moves to
    feature 0. A model trained only on the past (walk-forward) cannot anticipate
    the shift, while random K-fold sees samples from every period and quietly
    benefits from that future information.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    X = rng.normal(size=(n, n_features))
    w0 = 2.5 * (t / n)          # feature 0 matters late
    w1 = 2.5 * (1.0 - t / n)    # feature 1 matters early
    logit = X[:, 0] * w0 + X[:, 1] * w1 + 0.4 * X[:, 2]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def _cv_scores(X, y, splitter) -> np.ndarray:
    scores = []
    for train_idx, test_idx in splitter.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        scores.append(roc_auc_score(y[test_idx], proba))
    return np.array(scores)


def evaluate(n: int = 4000, n_splits: int = 5, seed: int = SEED) -> dict[str, Any]:
    X, y = make_drifting_timeseries(n=n, seed=seed)
    kfold = _cv_scores(X, y, StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    walk = _cv_scores(X, y, TimeSeriesSplit(n_splits=n_splits))
    return {
        "kfold_scores": kfold,
        "walk_scores": walk,
        "kfold_mean": float(kfold.mean()),
        "walk_mean": float(walk.mean()),
        "optimism": float(kfold.mean() - walk.mean()),
        "n_splits": n_splits,
    }


def main() -> None:
    r = evaluate()
    print(f"Random K-fold (shuffled)   AUC: {r['kfold_mean']:.3f} +/- {r['kfold_scores'].std():.3f}")
    print(f"Walk-forward (time-aware)  AUC: {r['walk_mean']:.3f} +/- {r['walk_scores'].std():.3f}")
    print(f"Optimism introduced by leakage: {r['optimism']:+.3f} AUC")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(1, r["n_splits"] + 1)
    ax.plot(x, r["kfold_scores"], "o-", color="#c0392b", label=f"random K-fold  (mean {r['kfold_mean']:.3f})")
    ax.plot(x, r["walk_scores"], "s-", color="#1f3a5f", label=f"walk-forward  (mean {r['walk_mean']:.3f})")
    ax.set_xlabel("fold")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Random K-fold overstates performance on time-series data")
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(IMAGES, "walk_forward_validation.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
