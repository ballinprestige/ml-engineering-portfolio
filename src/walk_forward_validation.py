"""Leakage-safe time-series validation.

Demonstrates why random K-fold cross-validation overstates performance on
time-ordered data, and how walk-forward (expanding-window) validation gives an
honest estimate.

The synthetic data has concept drift: the relationship between features and
target changes slowly over time. Random K-fold lets the model train on samples
temporally adjacent to the test samples (same regime), leaking information that
would not exist in production. Walk-forward validation always trains on the past
and tests on the future, matching how a model is actually deployed.
"""
import os

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


def make_drifting_timeseries(n=4000, n_features=10, seed=SEED):
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


def cv_scores(X, y, splitter):
    scores = []
    for train_idx, test_idx in splitter.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        scores.append(roc_auc_score(y[test_idx], proba))
    return np.array(scores)


def main():
    X, y = make_drifting_timeseries()
    n_splits = 5
    kfold = cv_scores(X, y, StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED))
    walk = cv_scores(X, y, TimeSeriesSplit(n_splits=n_splits))

    print(f"Random K-fold (shuffled)   AUC: {kfold.mean():.3f} +/- {kfold.std():.3f}")
    print(f"Walk-forward (time-aware)  AUC: {walk.mean():.3f} +/- {walk.std():.3f}")
    print(f"Optimism introduced by leakage: {kfold.mean() - walk.mean():+.3f} AUC")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(1, n_splits + 1)
    ax.plot(x, kfold, "o-", color="#c0392b", label=f"random K-fold  (mean {kfold.mean():.3f})")
    ax.plot(x, walk, "s-", color="#1f3a5f", label=f"walk-forward  (mean {walk.mean():.3f})")
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
