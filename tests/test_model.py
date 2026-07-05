"""Unit tests for saved model artifacts and inference (src/predict.py)"""

import numpy as np
import pandas as pd
import pytest

from src.utils import load_model, load_features, load_scaler, prepare_input, get_risk_level
from src.predict import predict

SAMPLE_PATIENT = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 130, "chol": 230,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2,
}


def test_artifacts_load():
    model = load_model()
    features = load_features()
    scaler = load_scaler()
    assert hasattr(model, "predict")
    assert len(features) == 13
    assert hasattr(scaler, "transform")


def test_prepare_input_scales_and_orders():
    features = load_features()
    X = prepare_input(SAMPLE_PATIENT)
    assert list(X.columns) == features
    # scaled values should not equal raw clinical values
    assert not np.isclose(X["chol"].iloc[0], SAMPLE_PATIENT["chol"])


def test_prepare_input_rejects_missing_features():
    bad = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
    with pytest.raises(ValueError, match="Missing features"):
        prepare_input(bad)


def test_predict_returns_valid_output():
    result = predict(SAMPLE_PATIENT)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ("low", "medium", "high")


def test_predict_is_deterministic():
    r1 = predict(SAMPLE_PATIENT)
    r2 = predict(SAMPLE_PATIENT)
    assert r1 == r2


@pytest.mark.parametrize("prob,expected", [
    (0.1, "low"), (0.29, "low"),
    (0.3, "medium"), (0.69, "medium"),
    (0.7, "high"), (0.95, "high"),
    (None, "unknown"),
])
def test_risk_level_bands(prob, expected):
    assert get_risk_level(prob) == expected
