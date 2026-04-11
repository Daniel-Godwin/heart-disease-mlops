"""
explainability.py
-----------------
Provides SHAP-based explanations for trained models.
Supports Logistic Regression, SVM, Random Forest.

Explain trained ML model using SHAP (TreeExplainer for Random Forest).

Outputs:
- Global feature importance (summary plot)
- Feature impact (bar plot)
- Local explanation for a single prediction
"""

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/features.pkl"
DATA_PATH = "data/processed/heart_processed.csv"   # adjust if needed
TARGET_COL = "target"

OUTPUT_DIR = "reports/shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD MODEL + DATA
# =========================
print("📦 Loading model and data...")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

df = pd.read_csv(DATA_PATH)

X = df[features]
y = df[TARGET_COL]


# =========================
# CREATE SHAP EXPLAINER
# =========================
print("🧠 Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# =========================
# GLOBAL EXPLANATION
# =========================
print("📊 Generating global SHAP summary plot...")

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary Plot (Global Feature Importance)")
plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", bbox_inches="tight")
plt.close()


# Bar plot (mean impact)
print("📊 Generating SHAP bar plot...")

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean Impact)")
plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", bbox_inches="tight")
plt.close()


# =========================
# LOCAL EXPLANATION (FIXED)
# =========================
print("🔍 Generating local explanation for one sample...")

sample_index = 0

# Create SHAP explanation object
shap_exp = explainer(X)

# For binary classification → select class 1 (positive class)
if len(shap_exp.values.shape) == 3:
    # case: (samples, features, classes)
    shap_values_single = shap_exp[sample_index, :, 1]
    base_value = shap_exp.base_values[sample_index, 1]
else:
    # fallback: already single output
    shap_values_single = shap_exp[sample_index]
    base_value = shap_exp.base_values[sample_index]

# Create explanation object manually (safe way)
single_explanation = shap.Explanation(
    values=shap_values_single,
    base_values=base_value,
    data=X.iloc[sample_index],
    feature_names=X.columns
)

shap.plots.waterfall(single_explanation)



# =========================
# SAVE SHAP VALUES (FIXED)
# =========================
print("💾 Saving SHAP values...")

import numpy as np

# shap_values is 3D: (samples, features, classes)
# Select class 1 (positive heart disease)
shap_values_class1 = shap_values[:, :, 1]

shap_df = pd.DataFrame(
    shap_values_class1,
    columns=features
)

shap_df.to_csv(f"{OUTPUT_DIR}/shap_values.csv", index=False)

print("✅ SHAP values saved successfully!")