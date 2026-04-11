import streamlit as st
import pandas as pd
import numpy as np
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


# =========================
# LOAD MODEL + DATA
# =========================
st.set_page_config(page_title="Clinical SHAP Dashboard", layout="wide")

st.title("🏥 Heart Disease Clinical AI Dashboard")
st.write("Explainable AI system using SHAP + Machine Learning")


model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

df = pd.read_csv(DATA_PATH)
X = df[features]


# =========================
# SHAP EXPLAINER
# =========================
@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(model)

explainer = load_explainer()
shap_exp = explainer(X)


# =========================
# SIDEBAR - PATIENT SELECT
# =========================
st.sidebar.header("Patient Selection")

patient_index = st.sidebar.slider(
    "Select patient index",
    0,
    len(X) - 1,
    0
)

patient_data = X.iloc[patient_index]


# =========================
# PREDICTION
# =========================
patient_df = pd.DataFrame([patient_data], columns=features)
prob = model.predict_proba(patient_df)[0][1]
prediction = "HIGH RISK" if prob > 0.5 else "LOW RISK"


# =========================
# DISPLAY RESULTS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Prediction Result")

    st.metric("Risk Probability", f"{prob:.2f}")
    st.markdown(f"### Status: {prediction}")

with col2:
    st.subheader("📊 Patient Data")
    st.dataframe(patient_df)


# =========================
# SHAP EXPLANATION
# =========================
st.subheader("🔍 SHAP Explainability (Feature Impact)")

shap_values = shap_exp.values[patient_index, :, 1]

shap_df = pd.DataFrame({
    "feature": features,
    "impact": shap_values
})

shap_df = shap_df.reindex(
    shap_df["impact"].abs().sort_values(ascending=False).index
)

st.bar_chart(shap_df.set_index("feature"))


# =========================
# TOP FEATURES
# =========================
st.subheader("⚠️ Top Contributing Factors")

top_features = shap_df.head(5)

for _, row in top_features.iterrows():
    direction = "increases" if row["impact"] > 0 else "decreases"
    st.write(f"• **{row['feature']}** {direction} risk")


# =========================
# PDF REPORT GENERATION
# =========================
def generate_pdf(index, prob, top_features):
    file_path = f"{OUTPUT_DIR}/patient_{index}_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    story = []

    text = f"""
    <b>Heart Disease Report</b><br/><br/>
    Risk Probability: {prob:.2f}<br/><br/>
    """

    for _, row in top_features.iterrows():
        direction = "increases" if row["impact"] > 0 else "decreases"
        text += f"- {row['feature']} {direction} risk<br/>"

    story.append(Paragraph(text, styles["Normal"]))
    story.append(Spacer(1, 12))

    doc.build(story)

    return file_path


# =========================
# DOWNLOAD REPORT BUTTON
# =========================
st.subheader("📄 Generate Clinical Report")

if st.button("Generate PDF Report"):
    path = generate_pdf(patient_index, prob, top_features)
    st.success(f"Report saved: {path}")