"""Run every demo end to end and regenerate all figures.

Usage:
    python run_all.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

DEMOS = [
    "walk_forward_validation",
    "probability_calibration",
    "shap_feature_selection",
    "drift_detection",
]


def main() -> None:
    for name in DEMOS:
        print(f"\n===== {name} =====")
        importlib.import_module(name).main()


if __name__ == "__main__":
    main()
