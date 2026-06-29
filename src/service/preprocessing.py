"""Shared preprocessing — must be importable so the saved pipeline can be unpickled."""

from __future__ import annotations

import numpy as np

FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]
# Columns where a 0 is physiologically impossible and really means "missing".
ZERO_AS_MISSING_IDX = [1, 2, 3, 4, 5]  # glucose, blood_pressure, skin_thickness, insulin, bmi


def recode_zeros(X):
    """Recode impossible zeros as NaN so the downstream imputer handles them.

    Defined at module scope (not a lambda) so it survives joblib (de)serialization.
    """
    X = np.asarray(X, dtype=float).copy()
    for j in ZERO_AS_MISSING_IDX:
        col = X[:, j]
        col[col == 0] = np.nan
        X[:, j] = col
    return X
