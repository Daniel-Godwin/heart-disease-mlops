"""
fairness_comparison.py
----------------------
Compare baseline vs fairness-aware model.
Outputs:
- Performance metrics
- Fairness metrics
- Comparison table
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


# ---------------------------
# LOAD DATA
# ---------------------------
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
TARGET_COL = "target"


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y, df


# ---------------------------
# FAIRNESS WEIGHTING
# ---------------------------
def compute_sample_weights(X, y, sensitive_col="sex"):
    df = X.copy()
    df["target"] = y

    group_target_counts = df.groupby([sensitive_col, "target"]).size()

    weights = []
    for _, row in df.iterrows():
        g = row[sensitive_col]
        t = row["target"]
        weights.append(1.0 / group_target_counts[(g, t)])

    weights = np.array(weights)
    return weights / weights.mean()


# ---------------------------
# FAIRNESS METRIC
# ---------------------------
def compute_group_metrics(df, y_true, y_pred, group_col):
    df = df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    results = []

    for g in df[group_col].unique():
        subset = df[df[group_col] == g]

        acc = accuracy_score(subset["y_true"], subset["y_pred"])
        rec = recall_score(subset["y_true"], subset["y_pred"])

        results.append({
            "group": g,
            "accuracy": acc,
            "recall": rec
        })

    results_df = pd.DataFrame(results)

    recall_gap = results_df["recall"].max() - results_df["recall"].min()
    acc_gap = results_df["accuracy"].max() - results_df["accuracy"].min()

    return results_df, recall_gap, acc_gap


# ---------------------------
# TRAIN MODELS
# ---------------------------
def train_models():
    X_train, y_train, df_train = load_data(TRAIN_PATH)
    X_test, y_test, df_test = load_data(TEST_PATH)

    # ---------------------------
    # BASELINE MODEL
    # ---------------------------
    baseline = RandomForestClassifier(random_state=42)
    baseline.fit(X_train, y_train)

    y_pred_base = baseline.predict(X_test)

    # ---------------------------
    # FAIR MODEL
    # ---------------------------
    weights = compute_sample_weights(X_train, y_train)

    fair_model = RandomForestClassifier(random_state=42)
    fair_model.fit(X_train, y_train, sample_weight=weights)

    y_pred_fair = fair_model.predict(X_test)

    # ---------------------------
    # PERFORMANCE
    # ---------------------------
    perf = pd.DataFrame([
        {
            "model": "baseline",
            "accuracy": accuracy_score(y_test, y_pred_base),
            "recall": recall_score(y_test, y_pred_base)
        },
        {
            "model": "fair_model",
            "accuracy": accuracy_score(y_test, y_pred_fair),
            "recall": recall_score(y_test, y_pred_fair)
        }
    ])

    # ---------------------------
    # FAIRNESS
    # ---------------------------
    base_fair, base_gap, _ = compute_group_metrics(df_test, y_test, y_pred_base, "sex")
    fair_fair, fair_gap, _ = compute_group_metrics(df_test, y_test, y_pred_fair, "sex")

    fairness_summary = pd.DataFrame([
        {"model": "baseline", "recall_gap": base_gap},
        {"model": "fair_model", "recall_gap": fair_gap}
    ])

    return perf, fairness_summary


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    perf, fairness = train_models()

    print("\n📊 PERFORMANCE COMPARISON:")
    print(perf)

    print("\n⚖️ FAIRNESS COMPARISON:")
    print(fairness)