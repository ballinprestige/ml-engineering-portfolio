"""Encode each demo's headline claim as an assertion.

Results are deterministic (fixed seeds), so these tests are stable in CI. Sizes
are reduced for speed; the relationships they assert hold at full scale too.
"""
import drift_detection as drift
import probability_calibration as cal
import shap_feature_selection as sfs
import walk_forward_validation as wfv


def test_kfold_overstates_on_timeseries():
    r = wfv.evaluate(n=3000, n_splits=5)
    assert r["kfold_mean"] > 0.6          # the model learns a real signal
    assert r["walk_mean"] > 0.5           # walk-forward still beats chance
    assert r["kfold_mean"] > r["walk_mean"]  # K-fold is optimistically high (leakage)


def test_calibration_improves_brier():
    r = cal.evaluate(n=5000, n_estimators=150)
    assert r["auc"] > 0.85
    assert r["brier_cal"] < r["brier_uncal"]   # calibration lowers Brier


def test_shap_pruning_preserves_auc():
    r = sfs.evaluate(n=4000, n_features=30, ks=(10,))
    assert r["top10_auc"] >= r["full_auc"] - 0.03   # 10 features ~= all 30


def test_drift_flags_shifted_features():
    r = drift.evaluate(n=4000)
    assert r["psi"][0] > drift.ALERT      # mean-shifted feature is flagged
    assert r["psi"][1] < drift.WATCH      # stable feature is not
    assert "feature_0" in r["flagged"]
