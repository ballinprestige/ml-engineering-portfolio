"""Enforce each demo's headline claim AND pin the README numbers.

These run the demos at their default (multi-seed) settings — the exact configuration the README
reports — and assert the documented values fall in tight bands. That makes the committed numbers
self-checking: if the code output drifts away from the README, CI fails.
"""

import drift_detection as drift
import probability_calibration as cal
import real_data_pipeline as rdp
import shap_feature_selection as sfs
import walk_forward_validation as wfv


def test_walk_forward_temporal_optimism():
    r = wfv.evaluate()
    assert r["trust_mean"] > 0.5
    assert r["leak_mean"] > r["trust_mean"]
    assert 0.005 < r["optimism_mean"] < 0.03  # README: +0.012
    assert 0.82 < r["leak_mean"] < 0.87
    assert 0.82 < r["trust_mean"] < 0.86


def test_calibration_lowers_brier_without_moving_ranking():
    r = cal.evaluate()
    assert r["brier_cal_mean"] < r["brier_uncal_mean"]
    assert r["auc_delta_max_abs"] < 0.005  # ranking near-unchanged (measured)
    assert 0.045 < r["brier_uncal_mean"] < 0.056  # README: 0.0503
    assert 0.97 < r["auc_uncal_mean"] < 0.985


def test_shap_selection_matches_full_on_untouched_test():
    r = sfs.evaluate()
    assert r["k_median"] < r["n_features"]
    assert abs(r["topk_test_mean"] - r["full_test_mean"]) < 0.01  # within noise on TEST
    assert 0.97 < r["full_test_mean"] < 0.99  # README: 0.981
    assert 6 <= r["k_median"] <= 12  # README: median 10


def test_drift_flags_every_shifted_feature():
    r = drift.evaluate()
    assert r["psi"][0] > drift.ALERT and r["psi"][2] > drift.ALERT  # shifted
    assert r["psi"][1] < drift.WATCH and r["psi"][3] < drift.WATCH and r["psi"][5] < drift.WATCH
    assert set(r["flagged"]) == {"feature_0", "feature_2"}
    assert 1.0 < r["psi"][0] < 1.5  # README: 1.26


def test_real_data_pipeline():
    r = rdp.evaluate()
    assert r["n_missing_total"] == 652  # deterministic: impossible zeros recoded as missing
    assert r["brier_cal_mean"] <= r["brier_uncal_mean"] + 0.001
    assert 0.78 < r["auc_uncal_mean"] < 0.85  # README: 0.817
