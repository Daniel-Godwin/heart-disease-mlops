"""
model_factory.py
----------------
Creates ML models in a modular + production-ready way.
Supports:
- Logistic Regression
- SVM
- Random Forest
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



def get_model(model_name: str):
    """
    Return initialized ML model based on name.
    """

    model_name = model_name.lower()

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=200,
            solver="liblinear",
            class_weight="balanced",
            random_state=42
        )

    elif model_name == "svm":
        return SVC(
            kernel="rbf",
            probability=True,   # IMPORTANT for explainability + API
            class_weight="balanced",
            random_state=42
        )

    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced"
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from "
            "['logistic_regression', 'svm', 'random_forest']"
        )