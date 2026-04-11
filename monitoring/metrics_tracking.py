"""
metrics_tracking.py
-------------------
Logs model predictions and performance drift.
"""

import json
import os
from datetime import datetime

LOG_FILE = "monitoring/logs.json"


def log_prediction(input_data, prediction, probability):
    os.makedirs("monitoring", exist_ok=True)

    log_entry = {
        "timestamp": str(datetime.now()),
        "input": input_data,
        "prediction": prediction,
        "probability": probability
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)