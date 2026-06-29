"""Run every demo end to end and regenerate all figures.

Requires the package installed: `pip install -e .`
Usage: python run_all.py
"""

import importlib

DEMOS = [
    "walk_forward_validation",
    "probability_calibration",
    "shap_feature_selection",
    "drift_detection",
    "real_data_pipeline",
]


def main() -> None:
    for name in DEMOS:
        print(f"\n===== {name} =====")
        importlib.import_module(name).main()


if __name__ == "__main__":
    main()
