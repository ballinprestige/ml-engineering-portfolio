"""Request/response schemas — a strict data contract (validation is the schema gate).

Bounds are deliberately conservative and tied to the training population (Pima women, age 21+).
`extra="forbid"` rejects unexpected fields, and integer types are enforced where appropriate so
the API cannot be fed nonsense and return a confident answer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_DISCLAIMER = (
    "Demonstration only — not medical advice. Trained on the Pima Indians Diabetes dataset "
    "(women of Pima heritage, age 21+); inputs or use outside that population are not valid."
)


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(ge=40, le=300)
    blood_pressure: float = Field(ge=20, le=200)
    skin_thickness: float = Field(ge=5, le=100)
    insulin: float = Field(ge=0, le=1000)
    bmi: float = Field(ge=10, le=70)
    diabetes_pedigree: float = Field(ge=0.0, le=3.0)
    age: int = Field(ge=21, le=100)


class Prediction(BaseModel):
    probability: float
    prediction: int
    threshold: float
    model_version: int
    disclaimer: str = _DISCLAIMER
