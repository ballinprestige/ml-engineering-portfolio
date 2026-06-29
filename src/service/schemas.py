"""Request/response schemas — the service's data contract (validation is the schema gate)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Features(BaseModel):
    pregnancies: float = Field(ge=0)
    glucose: float = Field(ge=0)
    blood_pressure: float = Field(ge=0)
    skin_thickness: float = Field(ge=0)
    insulin: float = Field(ge=0)
    bmi: float = Field(ge=0)
    diabetes_pedigree: float = Field(ge=0)
    age: float = Field(gt=0)


class Prediction(BaseModel):
    probability: float
    prediction: int
    model_version: int
