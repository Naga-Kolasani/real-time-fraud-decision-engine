"""
Append successful fraud-scoring events to a local JSONL log.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PREDICTION_LOG_PATH = Path("artifacts/logs/predictions.jsonl")


def log_prediction(transaction: dict[str, float], prediction: dict[str, Any]) -> None:
    """ Append one scored transaction and its result as one JSON line. """
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    prediction_timestamp = prediction["timestamp"]
    if hasattr(prediction_timestamp, "isoformat"):
        prediction_timestamp = prediction_timestamp.isoformat()

    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "transaction": transaction,
        "risk_score": prediction["risk_score"],
        "decision": prediction["decision"],
        "model_name": prediction["model_name"],
        "model_version": prediction["model_version"],
        "prediction_timestamp": prediction_timestamp,
    }

    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")