"""Probability calibration.

A model can rank well (high AUC) yet output miscalibrated probabilities: when it
says 0.9 it may be right only 70% of the time. For decision-making you need
probabilities you can trust. This demo trains a tree-ensemble classifier,
measures its calibration with a reliability diagram and the Brier score, then
applies isotonic calibration fit on a held-out calibration set and shows the
improvement. Note that AUC (ranking) is unchanged — calibration fixes the
*scale* of the probabilities, not their order.
"""
import os

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
SEED = 42


def main():
    X, y = make_classification(
        n_samples=14000, n_features=20, n_informative=6, n_redundant=4,
        weights=[0.7, 0.3], random_state=SEED,
    )
    # train / calibration / test (calibration set is held out from training)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)

    base = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=SEED, n_jobs=4)
    base.fit(X_tr, y_tr)

    p_uncal = base.predict_proba(X_te)[:, 1]
    # isotonic calibration fit on a held-out calibration set (base model frozen)
    calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrator.fit(X_cal, y_cal)
    p_cal = calibrator.predict_proba(X_te)[:, 1]

    print(f"AUC (unchanged by calibration): {roc_auc_score(y_te, p_uncal):.3f}")
    print(f"Brier  uncalibrated: {brier_score_loss(y_te, p_uncal):.4f}")
    print(f"Brier  calibrated:   {brier_score_loss(y_te, p_cal):.4f}")

    frac_u, mean_u = calibration_curve(y_te, p_uncal, n_bins=10, strategy="quantile")
    frac_c, mean_c = calibration_curve(y_te, p_cal, n_bins=10, strategy="quantile")

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
    out = os.path.join(IMAGES, "probability_calibration.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
