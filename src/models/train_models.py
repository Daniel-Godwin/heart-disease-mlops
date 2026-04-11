"""
train_models.py
---------------
Reusable training + evaluation functions.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score



def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """

    y_pred = model.predict(X_test)

    # Some models support probability
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

    return metrics


