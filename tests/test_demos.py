"""Enforce each demo's headline claim — including the ones the README states.

Sizes/seed-counts are reduced for CI speed; results are deterministic so the assertions are
stable. These test the actual claims (out-of-sample direction, AUC-near-invariance under
calibration, validation-selected SHAP matching full on an untouched test set), not just that
the scripts run.
"""

import drift_detection as drift
import probability_calibration as cal
import shap_feature_selection as sfs
import walk_forward_validation as wfv


def test_temporal_validation_optimism_is_positive():
    r = wfv.evaluate(n=3000, n_blocks=5, seeds=range(4))
    assert r["trust_mean"] > 0.5  # model learns a real signal
    assert r["leak_mean"] > r["trust_mean"]  # peeking at the test era inflates AUC
    assert r["optimism_mean"] > 0  # holds in aggregate across seeds


def test_calibration_lowers_brier_without_moving_ranking():
    r = cal.evaluate(n=6000, n_estimators=150, seeds=range(3))
    assert r["brier_cal_mean"] < r["brier_uncal_mean"]  # calibration helps
    assert r["auc_delta_max_abs"] < 0.005  # ranking near-unchanged (measured, not assumed)


def test_shap_selection_matches_full_on_untouched_test():
    r = sfs.evaluate(n=4000, n_features=30, seeds=range(3), grid=(5, 10, 15, 30))
    assert r["k_median"] < r["n_features"]  # genuinely fewer features
    assert abs(r["topk_test_mean"] - r["full_test_mean"]) < 0.02  # within noise on the TEST set


def test_drift_flags_shifted_features():
    r = drift.evaluate(n=4000)
    assert r["psi"][0] > drift.ALERT  # mean-shifted feature is flagged
    assert r["psi"][1] < drift.WATCH  # stable feature is not
    assert "feature_0" in r["flagged"]
