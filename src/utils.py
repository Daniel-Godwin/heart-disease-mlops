"""
utils.py
--------
Shared utilities for the heart disease MLOps pipeline:
- Centralized path configuration
- Artifact loading (model, scaler, feature names)
- Input preparation for inference (feature ordering + scaling)
"""

import os
import joblib
import pandas as pd

# -------------------------------
# PATHS (single source of truth)
# -------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "heart.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "heart_processed.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "processed", "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "processed", "test.csv")

MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

LOG_DIR = os.path.join(ROOT_DIR, "logs")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

TARGET_COL = "target"


# -------------------------------
# ARTIFACT LOADERS
# -------------------------------
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)


def load_features(path=FEATURES_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature list not found at {path}.")
    return joblib.load(path)


def load_scaler(path=SCALER_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}. Run preprocessing first.")
    return joblib.load(path)


# -------------------------------
# INFERENCE HELPERS
# -------------------------------
def prepare_input(input_data: dict, feature_names=None, scaler=None) -> pd.DataFrame:
    """
    Convert a raw patient dict into a scaled, correctly ordered DataFrame
    ready for model.predict().

    The model was trained on StandardScaler-transformed features, so raw
    clinical values MUST be scaled before prediction.
    """
    feature_names = feature_names if feature_names is not None else load_features()
    scaler = scaler if scaler is not None else load_scaler()

    X = pd.DataFrame([input_data])

    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    X = X[feature_names]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)
    return X_scaled


def get_risk_level(prob):
    """Map a probability into a clinical risk band."""
    if prob is None:
        return "unknown"
    if prob < 0.3:
        return "low"
    if prob < 0.7:
        return "medium"
    return "high"
