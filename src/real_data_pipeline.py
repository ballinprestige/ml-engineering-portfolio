"""Real-data pipeline — the same discipline on a messy public dataset.

Synthetic demos isolate one idea cleanly; real data is where the wrangling lives. This uses the
Pima Indians Diabetes dataset (UCI / OpenML), which has a classic real-world trap: several
columns record physiologically-impossible **zeros** that are really *missing* values
(insulin = 0, blood pressure = 0, ...). Naively treating them as real numbers poisons the model.

The pipeline:
1. recode the impossible zeros as missing,
2. impute + scale **inside a Pipeline fit on the training split only** (leakage-safe — the test
   set never informs the median used to impute it),
3. calibrate on a held-out validation split and report on an untouched test split,
reported as mean +/- std across seeds.

Data: Pima Indians Diabetes Database (originally UCI / NIDDK; via OpenML id 37), public.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "images")
DATA = os.path.join(ROOT, "data", "pima_diabetes.csv")
SEED = 42
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def load_data():
    """Load the dataset and recode impossible zeros as missing (the real-world wrangling step)."""
    df = pd.read_csv(DATA)
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    missing = {c: int(df[c].isna().sum()) for c in ZERO_AS_MISSING}
    return df, missing


def evaluate(seeds=range(5)) -> dict[str, Any]:
    df, missing = load_data()
    X = df.drop(columns="outcome").to_numpy()
    y = df["outcome"].to_numpy()

    au, ac, bu, bc = [], [], [], []
    rep = None
    for si, seed in enumerate(seeds):
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.4, random_state=seed, stratify=y
        )
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
        )
        # imputer + scaler are fit on TRAIN only -> the test set never leaks into preprocessing
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=seed)),
            ]
        )
        pipe.fit(X_tr, y_tr)
        p_uncal = pipe.predict_proba(X_te)[:, 1]
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(X_val, y_val)
        p_cal = cal.predict_proba(X_te)[:, 1]

        au.append(roc_auc_score(y_te, p_uncal))
        ac.append(roc_auc_score(y_te, p_cal))
        bu.append(brier_score_loss(y_te, p_uncal))
        bc.append(brier_score_loss(y_te, p_cal))
        if si == 0:
            rep = {
                "curve_uncal": calibration_curve(y_te, p_uncal, n_bins=8, strategy="quantile"),
                "curve_cal": calibration_curve(y_te, p_cal, n_bins=8, strategy="quantile"),
            }

    au, ac, bu, bc = map(np.array, (au, ac, bu, bc))
    return {
        "missing": missing,
        "n_missing_total": int(sum(missing.values())),
        "n_rows": int(len(df)),
        "auc_uncal_mean": float(au.mean()),
        "auc_uncal_std": float(au.std()),
        "auc_cal_mean": float(ac.mean()),
        "brier_uncal_mean": float(bu.mean()),
        "brier_cal_mean": float(bc.mean()),
        "n_seeds": len(list(seeds)),
        "rep": rep,
    }


def main() -> None:
    r = evaluate()
    print(
        f"rows: {r['n_rows']}  |  impossible-zero values recoded as missing: {r['n_missing_total']}"
    )
    for c, m in r["missing"].items():
        print(f"  {c:<16} missing: {m}")
    print(f"AUC (test): {r['auc_uncal_mean']:.3f} +/- {r['auc_uncal_std']:.3f}")
    print(
        f"Brier  uncalibrated: {r['brier_uncal_mean']:.4f}   calibrated: {r['brier_cal_mean']:.4f}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    cols = list(r["missing"])
    ax1.bar(cols, [r["missing"][c] for c in cols], color="#1f3a5f")
    ax1.set_title("Impossible zeros recoded as missing (real-world wrangling)")
    ax1.set_ylabel("missing values")
    ax1.tick_params(axis="x", rotation=30)

    (frac_u, mean_u) = r["rep"]["curve_uncal"]
    (frac_c, mean_c) = r["rep"]["curve_cal"]
    ax2.plot([0, 1], [0, 1], "--", color="#888", label="perfectly calibrated")
    ax2.plot(mean_u, frac_u, "o-", color="#c0392b", label="uncalibrated")
    ax2.plot(mean_c, frac_c, "s-", color="#1f3a5f", label="calibrated")
    ax2.set_title(f"Reliability on real test data (AUC {r['auc_uncal_mean']:.3f})")
    ax2.set_xlabel("mean predicted probability")
    ax2.set_ylabel("observed frequency")
    ax2.legend()
    fig.tight_layout()
    out = os.path.join(IMAGES, "real_data_pipeline.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
