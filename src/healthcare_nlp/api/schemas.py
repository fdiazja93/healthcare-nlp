"""Pydantic request/response schemas for the /predict endpoint."""
from __future__ import annotations
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """Payload accepted by POST /predict."""

    input_text: str

class PredictionResponse(BaseModel):
    """Response body returned by POST /predict."""

    probability: float
    label: int
    label_text: str
    model_name: str

