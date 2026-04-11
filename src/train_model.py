"""
train_model.py
Train selected ML model using factory pattern.
Logs results + saves model (MLflow integrated, production-ready).
"""

import mlflow
import mlflow.sklearn
import os
import joblib
import pandas as pd
import numpy as np



from src.models.model_factory import get_model
from src.models.train_models import evaluate_model
from mlflow.models.signature import infer_signature


# -------------------------------
# MLFLOW CONFIG
# -------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("heart_disease_experiment")


# -------------------------------
# PATH CONFIG
# -------------------------------
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_DIR = "models/"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")

TARGET_COL = "target"


# -------------------------------
# DATA LOADER
# -------------------------------
def load_data(path):
    data = pd.read_csv(path)

    if TARGET_COL not in data.columns:
        raise ValueError(f"❌ Target column '{TARGET_COL}' not found in dataset!")

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    return X, y


# -------------------------------
# FEATURE IMPORTANCE DEBUG (OPTIONAL BUT USEFUL)
# -------------------------------
def print_feature_importance(model, X):
    print("\n🔥 Feature Importance:")

    importances = None

    if hasattr(model, "feature_importances_"):  # Random Forest
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):  # Logistic Regression / SVM (linear)
        importances = np.abs(model.coef_[0])

    if importances is not None:
        feature_importance = sorted(
            zip(X.columns, importances),
            key=lambda x: x[1],
            reverse=True
        )

        for f, v in feature_importance:
            print(f"{f}: {v:.4f}")
    else:
        print("⚠️ Model does not support direct feature importance.")


# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train(model_name="random_forest"):
    print(f"🚀 Training model: {model_name}")

    # ---------------------------
    # 1. LOAD DATA
    # ---------------------------
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    print("✅ Training features:", X_train.columns.tolist())

    # ---------------------------
    # 2. INIT MODEL
    # ---------------------------
    model = get_model(model_name)

    # ---------------------------
    # 3. START MLFLOW RUN
    # ---------------------------
    with mlflow.start_run():

        # ---------------------------
        # 4. TRAIN
        # ---------------------------
        model.fit(X_train, y_train)

        # ---------------------------
        # 5. EVALUATE
        # ---------------------------
        train_metrics = evaluate_model(model, X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)

        print("\n📊 Train Metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")

        print("\n📊 Test Metrics:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        # ---------------------------
        # 6. LOG TO MLFLOW
        # ---------------------------
        mlflow.log_param("model_name", model_name)

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ---------------------------
        # 7. FEATURE IMPORTANCE (DEBUG)
        # ---------------------------
        print_feature_importance(model, X_train)

        # ---------------------------
        # 8. SAVE LOCALLY
        # ---------------------------
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(X_train.columns.tolist(), FEATURES_PATH)

        print(f"\n💾 Model saved at: {MODEL_PATH}")
        print(f"💾 Features saved at: {FEATURES_PATH}")

        # ---------------------------
        # 9. CREATE SIGNATURE (FIXED POSITION)
        # ---------------------------
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:1]

        # ---------------------------
        # 10. REGISTER MODEL IN MLFLOW
        # ---------------------------
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="heart_disease_model",
            signature=signature,
            input_example=input_example
        )

        print("✅ Model registered in MLflow")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    train("random_forest")