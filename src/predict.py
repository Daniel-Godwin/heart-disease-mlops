"""
predict.py
----------
Loads trained model and makes predictions for new inputs.
Production-ready inference module.

IMPORTANT: The model was trained on StandardScaler-transformed features,
so raw clinical inputs are scaled with the persisted scaler before
prediction (bug fix over the original version).
"""

from src.utils import load_model, load_features, load_scaler, prepare_input, get_risk_level


def predict(input_data: dict) -> dict:
    """Predict heart disease risk for one patient (raw clinical values)."""
    model = load_model()
    feature_names = load_features()
    scaler = load_scaler()

    X = prepare_input(input_data, feature_names=feature_names, scaler=scaler)

    prediction = model.predict(X)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(prob) if prob is not None else None,
        "risk_level": get_risk_level(prob),
    }


if __name__ == "__main__":
    sample = {
        "age": 52, "sex": 1, "cp": 0, "trestbps": 130, "chol": 230,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2,
    }
    print(predict(sample))
