"""Probability calibration.

A model can rank well (high AUC) yet output miscalibrated probabilities. Isotonic
calibration fit on a held-out set lowers the Brier score; because isotonic regression is
monotone it leaves the *ranking* nearly unchanged — but "nearly" is not "exactly" (ties at
step plateaus can perturb AUC slightly), so we MEASURE both AUCs rather than assert equality.
Reported as mean +/- std across seeds.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
SEEDS = range(5)


def evaluate(n: int = 14000, n_estimators: int = 300, seeds=SEEDS) -> dict[str, Any]:
    bu, bc, au, ac = [], [], [], []
    rep = None
    for si, seed in enumerate(seeds):
        X, y = make_classification(
            n_samples=n,
            n_features=20,
            n_informative=6,
            n_redundant=4,
            weights=[0.7, 0.3],
            random_state=seed,
        )
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.5, random_state=seed, stratify=y
        )
        X_cal, X_te, y_cal, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
        )

        base = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        base.fit(X_tr, y_tr)
        p_uncal = base.predict_proba(X_te)[:, 1]

        calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
        calibrator.fit(X_cal, y_cal)
        p_cal = calibrator.predict_proba(X_te)[:, 1]

        bu.append(brier_score_loss(y_te, p_uncal))
        bc.append(brier_score_loss(y_te, p_cal))
        au.append(roc_auc_score(y_te, p_uncal))
        ac.append(roc_auc_score(y_te, p_cal))
        if si == 0:
            rep = {
                "curve_uncal": calibration_curve(y_te, p_uncal, n_bins=10, strategy="quantile"),
                "curve_cal": calibration_curve(y_te, p_cal, n_bins=10, strategy="quantile"),
            }

    bu, bc, au, ac = map(np.array, (bu, bc, au, ac))
    return {
        "brier_uncal_mean": float(bu.mean()),
        "brier_uncal_std": float(bu.std()),
        "brier_cal_mean": float(bc.mean()),
        "brier_cal_std": float(bc.std()),
        "auc_uncal_mean": float(au.mean()),
        "auc_cal_mean": float(ac.mean()),
        "auc_delta_mean": float((ac - au).mean()),
        "auc_delta_max_abs": float(np.abs(ac - au).max()),
        "n_seeds": len(list(seeds)),
        "rep": rep,
    }


def main() -> None:
    r = evaluate()
    print(f"Brier uncalibrated: {r['brier_uncal_mean']:.4f} +/- {r['brier_uncal_std']:.4f}")
    print(f"Brier calibrated:   {r['brier_cal_mean']:.4f} +/- {r['brier_cal_std']:.4f}")
    print(f"AUC uncalibrated:   {r['auc_uncal_mean']:.4f}")
    print(
        f"AUC calibrated:     {r['auc_cal_mean']:.4f}  (mean change {r['auc_delta_mean']:+.5f}, max |change| {r['auc_delta_max_abs']:.5f})"
    )

    (frac_u, mean_u) = r["rep"]["curve_uncal"]
    (frac_c, mean_c) = r["rep"]["curve_cal"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfectly calibrated")
    ax.plot(mean_u, frac_u, "o-", color="#c0392b", label="uncalibrated")
    ax.plot(mean_c, frac_c, "s-", color="#1f3a5f", label="isotonic-calibrated")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram: before vs after calibration")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "probability_calibration.png"), dpi=130)
    print("saved", os.path.join(IMAGES, "probability_calibration.png"))


if __name__ == "__main__":
    main()
