"""Feature drift detection.

A deployed model silently degrades when the live data distribution drifts away
from what it was trained on. This demo compares a reference (training) sample
against a shifted "production" sample using two standard monitors:

* Population Stability Index (PSI) — < 0.1 stable, 0.1-0.25 watch, > 0.25 alert.
* Kolmogorov-Smirnov two-sample test — small p-value means the distributions differ.

`evaluate()` returns the metrics (importable + testable); `main()` adds the plots.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
SEED = 42
WATCH, ALERT = 0.1, 0.25


def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """PSI between a reference and current 1-D sample using reference quantile bins."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    cur_frac = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def make_reference_and_current(n: int = 6000, n_features: int = 6, seed: int = SEED):
    """Reference vs current data; features 0, 2, 4 drift, the rest are stable."""
    rng = np.random.default_rng(seed)
    ref = rng.normal(0, 1, (n, n_features))
    cur = rng.normal(0, 1, (n, n_features))
    cur[:, 0] += 1.2                          # feature 0: mean shift
    cur[:, 2] *= 1.8                          # feature 2: scale change
    cur[:, 4] = rng.normal(0.5, 1.0, n)       # feature 4: moderate mean shift
    names = [f"feature_{i}" for i in range(n_features)]
    return ref, cur, names


def evaluate(n: int = 6000, n_features: int = 6, seed: int = SEED) -> dict[str, Any]:
    ref, cur, names = make_reference_and_current(n=n, n_features=n_features, seed=seed)
    psi = np.array([population_stability_index(ref[:, i], cur[:, i]) for i in range(n_features)])
    ks_p = np.array([ks_2samp(ref[:, i], cur[:, i]).pvalue for i in range(n_features)])
    flagged = [names[i] for i in range(n_features) if psi[i] > ALERT]
    return {
        "names": names,
        "psi": psi,
        "ks_pvalues": ks_p,
        "flagged": flagged,
        "ref": ref,
        "cur": cur,
    }


def main() -> None:
    r = evaluate()
    print(f"{'feature':<12}{'PSI':>8}{'KS p-value':>14}   status")
    for i, name in enumerate(r["names"]):
        psi = r["psi"][i]
        status = "ALERT" if psi > ALERT else ("watch" if psi > WATCH else "stable")
        print(f"{name:<12}{psi:>8.3f}{r['ks_pvalues'][i]:>14.2e}   {status}")
    print("drift-flagged features:", ", ".join(r["flagged"]) or "none")

    colors = ["#c0392b" if p > ALERT else "#e0a64d" if p > WATCH else "#1f3a5f" for p in r["psi"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.bar(r["names"], r["psi"], color=colors)
    ax1.axhline(ALERT, ls="--", color="#c0392b", lw=1, label=f"alert ({ALERT})")
    ax1.axhline(WATCH, ls="--", color="#e0a64d", lw=1, label=f"watch ({WATCH})")
    ax1.set_ylabel("PSI")
    ax1.set_title("Population Stability Index by feature")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend()

    worst = int(np.argmax(r["psi"]))
    ax2.hist(r["ref"][:, worst], bins=40, alpha=0.6, color="#1f3a5f", label="reference", density=True)
    ax2.hist(r["cur"][:, worst], bins=40, alpha=0.6, color="#c0392b", label="current", density=True)
    ax2.set_title(f"Most-drifted feature: {r['names'][worst]} (PSI {r['psi'][worst]:.2f})")
    ax2.set_xlabel("value")
    ax2.set_ylabel("density")
    ax2.legend()
    fig.tight_layout()
    out = os.path.join(IMAGES, "drift_detection.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
