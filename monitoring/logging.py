"""
logging.py
----------
Structured prediction logging for production monitoring.

Every prediction served by the API (or batch jobs) is appended as a JSON
line to logs/predictions.jsonl. These logs feed drift detection and
performance monitoring downstream.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone

from src.utils import LOG_DIR

PREDICTION_LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")

logger = logging.getLogger("heart_disease_mlops")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)


def log_prediction(input_data: dict, prediction: int, probability: float,
                   risk_level: str, model_version: str = "v1",
                   log_path: str = None) -> dict:
    """Append a structured prediction record to the JSONL log."""
    log_path = log_path or PREDICTION_LOG_PATH
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    record = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "input": input_data,
        "prediction": int(prediction),
        "probability": round(float(probability), 4) if probability is not None else None,
        "risk_level": risk_level,
    }

    with open(log_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")

    logger.info("prediction=%s prob=%s risk=%s id=%s",
                record["prediction"], record["probability"],
                record["risk_level"], record["request_id"])
    return record


def load_prediction_logs(log_path: str = None) -> list:
    """Read all logged predictions (for drift analysis on served traffic)."""
    log_path = log_path or PREDICTION_LOG_PATH
    if not os.path.exists(log_path):
        return []
    with open(log_path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
