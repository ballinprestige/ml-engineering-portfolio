"""Leakage-safe time-series validation (a controlled comparison).

A fair test of "does training on non-future data matter?" must hold the training-set SIZE
fixed, or the result confounds leakage with how much data each scheme saw. So for each
time-ordered test block we train two models on the SAME number of rows (W):

* time-respecting: the W rows immediately BEFORE the test block (deployment-realistic).
* leakage: W rows from the test block's own ERA — the window straddling it (rows just before
  AND just after, i.e. peeking at the future). Same size W.

Under smooth linear drift the straddling window is centered on the test block's regime and sees
future-adjacent rows, so it reports an optimistic AUC; the time-respecting model, trained only on
the slightly-staler past, is honest. We report the gap ("temporal-validation optimism") as
mean +/- std across seeds — not a single point estimate, and matched training sizes isolate the
cause to *which rows are allowed*, not how many.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
SEEDS = range(10)


def make_drifting_timeseries(n: int = 4000, n_features: int = 10, seed: int = 0):
    """Time-ordered data with smooth linear drift: the signal moves from feature 1 to feature 0."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    X = rng.normal(size=(n, n_features))
    w0 = 2.5 * (t / n)  # feature 0 matters late
    w1 = 2.5 * (1.0 - t / n)  # feature 1 matters early
    logit = X[:, 0] * w0 + X[:, 1] * w1 + 0.4 * X[:, 2]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def _auc(X_tr, y_tr, X_te, y_te) -> float:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    return roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])


def evaluate(n: int = 4000, n_features: int = 10, n_blocks: int = 5, seeds=SEEDS) -> dict[str, Any]:
    leak, trust = [], []
    rep = None
    for si, seed in enumerate(seeds):
        X, y = make_drifting_timeseries(n, n_features, seed)
        rng = np.random.default_rng(seed + 1000)
        block = n // (n_blocks + 1)
        W = block  # matched training-set size for both schemes
        leak_f, trust_f = [], []
        for i in range(1, n_blocks + 1):
            ts, te = i * block, i * block + block
            test = np.arange(ts, te)
            past = np.arange(ts - W, ts)  # time-respecting (past only)
            local = np.concatenate(
                [past, np.arange(te, min(n, te + W))]
            )  # straddles the test block,
            leak_idx = rng.choice(
                local, size=W, replace=len(local) < W
            )  # incl. future-adjacent rows
            trust_f.append(_auc(X[past], y[past], X[test], y[test]))
            leak_f.append(_auc(X[leak_idx], y[leak_idx], X[test], y[test]))
        leak.append(float(np.mean(leak_f)))
        trust.append(float(np.mean(trust_f)))
        if si == 0:
            rep = {"leak": leak_f, "trust": trust_f, "n_blocks": n_blocks}

    leak, trust = np.array(leak), np.array(trust)
    opt = leak - trust
    return {
        "leak_mean": float(leak.mean()),
        "leak_std": float(leak.std()),
        "trust_mean": float(trust.mean()),
        "trust_std": float(trust.std()),
        "optimism_mean": float(opt.mean()),
        "optimism_std": float(opt.std()),
        "n_seeds": len(list(seeds)),
        "rep": rep,
    }


def main() -> None:
    r = evaluate()
    print(
        f"Leakage (test-era incl. future)       AUC: {r['leak_mean']:.3f} +/- {r['leak_std']:.3f}"
    )
    print(
        f"Time-respecting (past rows only)      AUC: {r['trust_mean']:.3f} +/- {r['trust_std']:.3f}"
    )
    print(
        f"Temporal-validation optimism: {r['optimism_mean']:+.3f} +/- {r['optimism_std']:.3f} AUC (over {r['n_seeds']} seeds)"
    )

    rep = r["rep"]
    x = np.arange(1, rep["n_blocks"] + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        x,
        rep["leak"],
        "o-",
        color="#c0392b",
        label=f"leakage (test-era + future)  mean {r['leak_mean']:.3f}",
    )
    ax.plot(
        x,
        rep["trust"],
        "s-",
        color="#1f3a5f",
        label=f"time-respecting (past)  mean {r['trust_mean']:.3f}",
    )
    ax.set_xlabel("time-ordered test block")
    ax.set_ylabel("ROC AUC")
    ax.set_title(
        f"Matched training-set size; only the row-selection differs\n"
        f"optimism {r['optimism_mean']:+.3f} ± {r['optimism_std']:.3f} AUC over {r['n_seeds']} seeds",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "walk_forward_validation.png"), dpi=130)
    print("saved", os.path.join(IMAGES, "walk_forward_validation.png"))


if __name__ == "__main__":
    main()
