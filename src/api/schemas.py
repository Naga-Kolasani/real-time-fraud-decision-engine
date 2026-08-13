"""
Request/response shapes for `/score_transaction` and `/health`.

No app, routes, or model loading here - just the schemas.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator


class TransactionRequest(BaseModel):
    """
    One transaction. Time and Amount are raw values - scaling happens in the pipeline, not here. V1-V28 are the anonymized PCA features from the dataset.
    """

    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    @field_validator("*")
    @classmethod
    def check_finite(cls, value: float, info) -> float:
        # Same check for all 30 fields since they're all floats.
        # Reject NaN and infinity before they reach the model.
        if not math.isfinite(value):
            raise ValueError(
                f"{info.field_name} must be a finite number, got {value}"
            )
        return value


class ScoreResponse(BaseModel):
    """
    Matches what score_transaction() returns. No top_features yet - get_top_features() isn't implemented.
    """

    risk_score: float
    decision: Literal["approve", "review", "block"]
    model_name: str
    model_version: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """
    Just enough to confirm the model is loaded. Keeping versioning simple for now.
    """

    status: Literal["ok"]
    model_name: str
    model_version: str
