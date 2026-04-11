import mlflow
import mlflow.sklearn
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

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
        raise ValueError(f"❌ Target column '{TARGET_COL}' not found!")

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    return X, y


# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
def print_feature_importance(model, X):
    print("\n🔥 Feature Importance:")

    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is not None:
        for f, v in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
            print(f"{f}: {v:.4f}")
    else:
        print("⚠️ No feature importance available.")


# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train(model_name="random_forest"):
    print(f"🚀 Training model: {model_name}")

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    print("✅ Training features:", X_train.columns.tolist())

    # ---------------------------
    # INIT MODEL
    # ---------------------------
    base_model = get_model(model_name)

    # ---------------------------
    # HYPERPARAMETER SEARCH (ONLY RF)
    # ---------------------------
    if model_name == "random_forest":
        print("🧠 Running Hyperparameter Tuning...")

        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,
            scoring="roc_auc",
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

    else:
        model = base_model

    # ---------------------------
    # MLFLOW RUN
    # ---------------------------
    with mlflow.start_run():

        # ---------------------------
        # TRAIN
        # ---------------------------
        model.fit(X_train, y_train)

        # ---------------------------
        # HANDLE BEST MODEL
        # ---------------------------
        if isinstance(model, RandomizedSearchCV):
            best_model = model.best_estimator_

            print("✅ Best Parameters:", model.best_params_)
            print(f"📊 Best CV Score: {model.best_score_:.4f}")

            mlflow.log_params(model.best_params_)
            mlflow.log_metric("cv_score", model.best_score_)

        else:
            best_model = model

        # ---------------------------
        # EVALUATE
        # ---------------------------
        train_metrics = evaluate_model(best_model, X_train, y_train)
        test_metrics = evaluate_model(best_model, X_test, y_test)

        print("\n📊 Train Metrics:")
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")

        print("\n📊 Test Metrics:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        # ---------------------------
        # LOG METRICS
        # ---------------------------
        mlflow.log_param("model_name", model_name)

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ---------------------------
        # FEATURE IMPORTANCE
        # ---------------------------
        print_feature_importance(best_model, X_train)

        # ---------------------------
        # SAVE MODEL
        # ---------------------------
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(X_train.columns.tolist(), FEATURES_PATH)

        print(f"\n💾 Model saved at: {MODEL_PATH}")

        # ---------------------------
        # SIGNATURE
        # ---------------------------
        signature = infer_signature(X_train, best_model.predict(X_train))
        input_example = X_train.iloc[:1]

        # ---------------------------
        # REGISTER MODEL
        # ---------------------------
        mlflow.sklearn.log_model(
            best_model,
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