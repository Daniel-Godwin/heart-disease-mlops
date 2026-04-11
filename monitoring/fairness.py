"""
fairness.py
-----------
Evaluates model fairness across sensitive attributes (e.g., sex, age).
"""

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score


# =========================
# CONFIG
# =========================
MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/processed/heart_processed.csv"
TARGET_COL = "target"

SENSITIVE_FEATURES = ["sex"]  # you can add "age_group" later


# =========================
# LOAD
# =========================
print("📦 Loading model and data...")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# =========================
# CREATE AGE GROUP (OPTIONAL)
# =========================
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 55, 100],
    labels=["Young", "Middle", "Old"]
)


# =========================
# FAIRNESS FUNCTION
# =========================
def evaluate_group_performance(feature):

    print(f"\n⚖️ Fairness Analysis for: {feature}")

    groups = df[feature].unique()

    results = []

    for g in groups:
        idx = df[feature] == g

        X_group = X[idx]
        y_group = y[idx]

        if len(y_group) < 10:
            continue  # avoid unreliable small groups

        y_pred = model.predict(X_group)

        acc = accuracy_score(y_group, y_pred)
        rec = recall_score(y_group, y_pred)
        prec = precision_score(y_group, y_pred)

        results.append({
            "group": g,
            "size": len(y_group),
            "accuracy": acc,
            "recall": rec,
            "precision": prec
        })

    results_df = pd.DataFrame(results)

    # =========================
    # BIAS GAP
    # =========================
    recall_gap = results_df["recall"].max() - results_df["recall"].min()
    acc_gap = results_df["accuracy"].max() - results_df["accuracy"].min()

    print("\n📊 Group Metrics:")
    print(results_df)

    print("\n⚠️ Fairness Gaps:")
    print(f"Recall Gap: {recall_gap:.4f}")
    print(f"Accuracy Gap: {acc_gap:.4f}")

    # =========================
    # INTERPRETATION
    # =========================
    if recall_gap > 0.1:
        print("❗ Potential bias detected (recall gap > 0.1)")
    else:
        print("✅ Model appears reasonably fair (recall gap small)")

    return results_df


# =========================
# RUN
# =========================
if __name__ == "__main__":

    for feature in ["sex", "age_group"]:
        evaluate_group_performance(feature)