# ML Engineering Portfolio — Kyle Reynolds

![CI](https://github.com/ballinprestige/ml-engineering-portfolio/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)

A small **methods toolkit** for the production-ML concerns that decide whether a model that
looks good offline actually holds up live: leakage-safe validation, probability calibration,
honest feature selection, and drift monitoring. Every demo is self-contained, deterministic, and
its **headline claim is enforced by a test in CI** — measured out-of-sample, reported as mean ±
std across seeds, not a single lucky number.

> Scope note: four demos isolate a single concept on **synthetic benchmarks**; one runs
> end-to-end on a **messy real public dataset** (with the wrangling that implies). These are
> methodology demonstrations, not a deployed service — the point is correct evaluation protocol.

## Contents
- [Production pipeline (reference architecture)](#production-pipeline-reference-architecture)
- [1 · Leakage-safe time-series validation](#1--leakage-safe-time-series-validation)
- [2 · Probability calibration](#2--probability-calibration)
- [3 · SHAP-driven feature selection](#3--shap-driven-feature-selection)
- [4 · Feature drift detection](#4--feature-drift-detection)
- [5 · Real-data pipeline](#5--real-data-pipeline-messy-public-data)
- [Notebook walkthrough](#notebook-walkthrough) · [Run it](#run-it)

---

## Production pipeline (reference architecture)

```mermaid
flowchart LR
    A["Ingestion<br/>APIs and feeds"] --> B["Schema / data-contract<br/>checks"]
    B --> C["Feature engineering"]
    C --> D[("Feature store")]
    D --> E["Training"]
    E --> F["Validation<br/>(walk-forward, leakage audit,<br/>calibration)"]
    F --> G{"Promotion gate<br/>shadow compare, rollback"}
    G --> H["Serving"]
    H --> I["Drift / data-health<br/>monitoring"]
    I -->|"alert: investigate / retrain"| C
```

Schema checks gate the data *before* training; calibration is part of *validation*; shadow
comparison and rollback are one *promotion gate*; drift monitoring closes the loop.

---

## 1 · Leakage-safe time-series validation

How you split time-ordered data changes the estimate you get. This is a **controlled**
comparison — both models train on the **same number of rows (W)**, so the only difference is
*which* rows are allowed:

- **time-respecting** — the W rows immediately *before* each test block (deployment-realistic).
- **leakage** — W rows from the test block's own era, *including future-adjacent rows*.

Under smooth linear drift, peeking at the test era inflates the score. Reported across 10 seeds:

| Training rows (matched size W) | ROC AUC |
|---|---|
| Leakage (test-era + future) | **0.852 ± 0.005** |
| Time-respecting (past only) | **0.840 ± 0.005** |
| **Temporal-validation optimism** | **+0.012 ± 0.002** |

A small but robust effect — and it's isolated to *row selection*, not training-set size.

![Walk-forward vs leakage](images/walk_forward_validation.png)
→ [`src/walk_forward_validation.py`](src/walk_forward_validation.py)

---

## 2 · Probability calibration

A model can rank well yet be miscalibrated. Isotonic calibration (fit on a held-out set) lowers
the Brier score; because it's monotone the ranking is *nearly* unchanged — and we **measure**
that rather than assert it. Across 5 seeds:

| Metric | Uncalibrated | Isotonic-calibrated |
|---|---|---|
| Brier (lower better) | 0.0503 ± 0.0034 | **0.0454 ± 0.0032** |
| ROC AUC | 0.9779 | 0.9774 |

AUC change: mean **−0.0005**, max single-seed **0.0011** — small, real, and not zero (isotonic
ties can nudge ranking). A test asserts `|Δauc| < 0.005`.

![Reliability diagram](images/probability_calibration.png)
→ [`src/probability_calibration.py`](src/probability_calibration.py)

---

## 3 · SHAP-driven feature selection

Selecting features *and the count k* is a decision — made on the **validation** set, never the
test set. Protocol: fit on **train**, rank by mean |SHAP| and choose k on **validation**, report
once on an **untouched test** set, repeated across 10 seeds:

| Feature set | Test ROC AUC |
|---|---|
| All 40 features | 0.981 ± 0.007 |
| Validation-selected (median **k = 10**) | **0.981 ± 0.006** |

The selected ~10 features **match the full 40-feature model on held-out test data within noise**
— ~4× fewer *input features* (not a claim about model size or inference cost).

![SHAP selection on validation](images/shap_feature_selection.png)
![SHAP importance](images/shap_importance.png)
→ [`src/shap_feature_selection.py`](src/shap_feature_selection.py)

---

## 4 · Feature drift detection

When live data drifts from training data a model can silently degrade. Population Stability
Index (PSI) and the Kolmogorov–Smirnov test flag *which* features shifted:

| Feature | PSI | Status |
|---|---|---|
| feature_0 (mean shift) | 1.26 | **alert** |
| feature_2 (scale change) | 0.37 | **alert** |
| feature_4 (moderate shift) | 0.24 | watch |
| stable features | < 0.01 | ok |

Caveats kept honest: PSI's 0.1 / 0.25 bands are **heuristics, not laws**; the KS **p-value** is
sample-size-driven (≈0 for any real shift at large n) and needs multiple-testing correction, so
the demo also reports the KS **D-statistic** (an effect size). And **input drift is an
early-warning signal, not proof of model degradation** — it triggers investigation/retraining,
it doesn't by itself mean accuracy dropped.

![Drift detection](images/drift_detection.png)
→ [`src/drift_detection.py`](src/drift_detection.py)

---

## 5 · Real-data pipeline (messy public data)

Synthetic demos isolate one idea; real data is where the wrangling lives. On the **Pima Indians
Diabetes** dataset (UCI / OpenML), several columns record physiologically-impossible **zeros**
that are really *missing* (insulin = 0, blood pressure = 0, …) — a classic trap. The pipeline
recodes them, then **imputes and scales inside a `Pipeline` fit on the training split only** (so
the test set never informs its own imputation), and calibrates on validation before reporting on
an untouched test split:

| | value |
|---|---|
| Impossible-zero values recoded as missing | **652** (across 768 rows) |
| Test ROC AUC | 0.817 ± 0.030 |
| Brier (uncalibrated → calibrated) | 0.1708 → 0.1683 |

![Real-data pipeline](images/real_data_pipeline.png)
→ [`src/real_data_pipeline.py`](src/real_data_pipeline.py) · data: [`data/pima_diabetes.csv`](data/pima_diabetes.csv)

---

## Notebook walkthrough
A narrative version with the four synthetic demos:
[`notebooks/portfolio_walkthrough.ipynb`](notebooks/portfolio_walkthrough.ipynb)

## Run it
```bash
pip install -r requirements.txt
python run_all.py          # run all four demos, regenerate every figure

# development
pip install -r requirements-dev.txt
pytest -q                  # each demo's headline result is asserted here
ruff check src tests && ruff format --check src tests
```
Results are deterministic per seed and reported across seeds. CI runs the tests, all demos, and
the notebook on every push.

## About
Self-taught machine learning engineer focused on production-ML methods — leakage-safe
validation, calibration, feature selection, and drift monitoring.

**Stack:** Python · scikit-learn · XGBoost · SHAP · SciPy · pandas · NumPy · matplotlib

📫 kyleandgeorgi@gmail.com · [LinkedIn](https://www.linkedin.com/in/kyle-reynolds-29865423a/)
