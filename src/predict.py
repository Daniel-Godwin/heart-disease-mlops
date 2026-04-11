"""
predict.py
----------
Loads trained model and makes predictions for new inputs.
Production-ready inference module.
"""

import joblib
import pandas as pd
import os

MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/features.pkl"



def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train model first.")

    return joblib.load(MODEL_PATH)


def load_features():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Feature file not found.")

    return joblib.load(FEATURES_PATH)

def get_risk_level(prob):
    if prob is None:
        return "unknown"
    elif prob < 0.3:
        return "low"
    elif prob < 0.7:
        return "medium"
    else:
        return "high"
    
def predict(input_data: dict):

    model = load_model()
    feature_names = load_features()

    # Convert input to DataFrame
    X = pd.DataFrame([input_data])

    # ✅ Ensure all required features exist
    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # ✅ Ensure correct order
    X = X[feature_names]

    prediction = model.predict(X)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(prob) if prob is not None else None,
         "risk_level": get_risk_level(prob)
    }


if __name__ == "__main__":
    sample = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 230,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }

    print(predict(sample))