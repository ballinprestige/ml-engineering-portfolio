# Model Card — Pima Diabetes demonstration model

> **Demonstration only. Not medical advice; not for clinical, screening, or individual
> decision-making.** This model exists to demonstrate ML-engineering practice (leakage-safe,
> calibrated, monitored, served), not to predict anyone's health.

## Overview
- **Task:** binary classification of the diabetes `outcome` label in the Pima dataset.
- **Model:** scikit-learn `GradientBoostingClassifier` wrapped in **sigmoid (Platt) probability
  calibration** (sigmoid, not isotonic, because the calibration split is small), served as one
  end-to-end pipeline: recode impossible zeros → median impute → standardize → gradient boosting →
  calibration. All preprocessing is fit on the training split.
- **Artifact:** versioned and SHA-256-checksummed in `models/registry/`; full lineage in
  `metadata.json` (exposed at `GET /model`).

## Intended use / out of scope
- **Intended:** a portfolio demonstration of correct ML evaluation and a fail-fast serving layer.
- **Out of scope:** any real medical use, risk assessment, or decision about a real person.

## Data & population
- **Source:** Pima Indians Diabetes Database (UCI / OpenML id 37), public.
- **Population:** women of Pima heritage, **age ≥ 21**. The model is only meaningful within this
  population. The API enforces `age ≥ 21` and conservative physiological bounds, but inputs that
  are merely *in range* yet outside the training population are still **not valid**.
- **Data quirk:** several columns encode physiologically-impossible zeros (insulin, blood
  pressure, skin thickness, glucose, BMI) that are really *missing*; these are recoded to missing
  and imputed with **training-set medians only**.

## Metrics
- Reported on a held-out test split (seed 42; 70/15/15 train/calibration/test). Both **baseline
  (uncalibrated)** and **sigmoid-calibrated** AUC + Brier are recorded in the artifact metadata.
- For seed 42: AUC **0.822** (unchanged by sigmoid calibration — ranking preserved), Brier
  improved **0.176 → 0.170**. This is **one draw**; the README demo's **0.817 ± 0.030** is a
  cross-split *variability estimate*, not a formal confidence interval and not this artifact's number.

## Decision threshold
- Default **0.5** — arbitrary, and recorded in metadata and returned with every prediction. It
  **should be tuned to the relative cost of false positives vs. false negatives**; 0.5 is only a
  placeholder for a demo.

## Limitations & fairness
- Small dataset (768 rows) → noisy estimates; not enough data for strong claims.
- Single demographic group → **no generalization** beyond it; not evaluated for subgroup fairness.
- No live-input drift monitoring in the service (the drift demo is offline/synthetic).
- Not load-tested, not security-reviewed, not for production decisions.
