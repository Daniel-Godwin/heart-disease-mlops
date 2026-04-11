"""
clinical_shap_report.py
------------------------
Generates clinical SHAP explanation report for heart disease prediction.

Outputs:
- Patient SHAP explanation (top features)
- Clinical interpretation text
- PDF report
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# CONFIG
# =========================
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/features.pkl"
DATA_PATH = "data/processed/heart_processed.csv"

OUTPUT_DIR = "reports/clinical"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD MODEL + DATA
# =========================
print("📦 Loading model and data...")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

df = pd.read_csv(DATA_PATH)

X = df[features]


# =========================
# SHAP EXPLAINER
# =========================
print("🧠 Building SHAP explainer...")

explainer = shap.TreeExplainer(model)

# Convert once (IMPORTANT: SHAP v0.20+ safe format)
shap_exp = explainer(X)

# =========================
# CLINICAL REPORT FUNCTION
# =========================
def generate_clinical_report(patient_index=0):

    print("🔍 Generating clinical report...")

    # -------------------------
    # Patient data
    # -------------------------
    patient_data = X.iloc[patient_index]

    # -------------------------
    # Prediction (FIXED warning-safe)
    # -------------------------
    patient_df = pd.DataFrame([patient_data], columns=features)
    prob = model.predict_proba(patient_df)[0][1]

    # -------------------------
    # SHAP values (FIXED for binary classification)
    # -------------------------
    shap_vals = shap_exp.values[patient_index, :, 1]  # class 1 only

    # -------------------------
    # Feature importance (FIXED sorting)
    # -------------------------
    feature_impacts = pd.DataFrame({
        "feature": features,
        "shap_value": shap_vals.astype(float)
    })

    # sort by absolute impact
    feature_impacts = feature_impacts.loc[
        feature_impacts["shap_value"].abs().sort_values(ascending=False).index
    ]

    top_features = feature_impacts.head(5)

    # =========================
    # CLINICAL INTERPRETATION
    # =========================
    interpretation = f"""
    <b>Heart Disease Risk Report</b><br/><br/>

    <b>Predicted Risk Probability:</b> {prob:.2f}<br/><br/>

    <b>Top Contributing Factors:</b><br/>
    """

    for _, row in top_features.iterrows():
        direction = "increases" if row["shap_value"] > 0 else "decreases"
        interpretation += f"- {row['feature']} {direction} risk<br/>"

    interpretation += "<br/><b>Clinical Summary:</b><br/>"

    if prob > 0.7:
        interpretation += "High risk of heart disease. Immediate clinical attention recommended."
    elif prob > 0.4:
        interpretation += "Moderate risk. Further diagnostic tests advised."
    else:
        interpretation += "Low risk. Routine monitoring recommended."

    # =========================
    # CREATE PDF
    # =========================
    file_path = f"{OUTPUT_DIR}/patient_{patient_index}_report.pdf"
    doc = SimpleDocTemplate(file_path)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(interpretation, styles["Normal"]))
    story.append(Spacer(1, 12))

    doc.build(story)

    print(f"✅ Report saved: {file_path}")

    return file_path


# =========================
# RUN EXAMPLE
# =========================
if __name__ == "__main__":
    generate_clinical_report(patient_index=0)