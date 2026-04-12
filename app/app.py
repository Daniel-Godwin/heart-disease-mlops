import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from src.explainability.shap_engine import SHAPEngine
from src.fairness.fairness_engine import FairnessEngine
from src.reports.clinical_report_engine import ClinicalReportEngine


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Clinical Decision Support System",
    layout="wide",
    page_icon="🏥"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main { background-color: #f6f8fc; }

.risk-card {
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-weight: bold;
    text-align: center;
    font-size: 18px;
}

.high { background-color: #e74c3c; }
.medium { background-color: #f39c12; }
.low { background-color: #2ecc71; }

.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("""
# 🏥 Clinical Decision Support System
### AI-powered Heart Disease Prediction with Explainability & Fairness
---
""")


# =========================
# PATHS
# =========================
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/features.pkl"
DATA_PATH = "data/processed/heart_processed.csv"
OUTPUT_DIR = "reports/clinical"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    return joblib.load(FEATURES_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
features = load_features()
df = load_data()

# IMPORTANT: define age group FIRST
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 55, 100],
    labels=["Young", "Middle", "Old"]
)

X = df[features]

# =========================
# INIT ENGINES
# =========================

shap_engine = SHAPEngine(model, X, features)

fairness_engine = FairnessEngine(
    model=model,
    X=X,
    y=df["target"],
    features=features,
    raw_df=df
)

report_engine = ClinicalReportEngine()


# =========================
# SIDEBAR (IMPORTANT ORDER FIX)
# =========================
st.sidebar.header("👤 Patient Selection")

patient_index = st.sidebar.slider(
    "Select Patient ID",
    0,
    len(X) - 1,
    0
)

patient_data = X.iloc[[patient_index]]


# =========================
# PREDICTION
# =========================
prob = model.predict_proba(patient_data)[0][1]
prediction = model.predict(patient_data)[0]

risk_label = "🔴 HIGH RISK" if prob > 0.7 else "🟡 MODERATE RISK" if prob > 0.4 else "🟢 LOW RISK"
risk_class = "high" if prob > 0.7 else "medium" if prob > 0.4 else "low"


# =========================
# UI CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risk Probability", f"{prob:.2f}")

with col2:
    st.markdown(f"<div class='risk-card {risk_class}'>{risk_label}</div>", unsafe_allow_html=True)

with col3:
    st.metric("Prediction", "Positive" if prediction == 1 else "Negative")


st.subheader("📋 Patient Clinical Profile")
st.dataframe(patient_data)


# =========================
# SHAP (FIXED - USING ENGINE ONLY)
# =========================
st.subheader("🧠 Explainability (SHAP)")

explanation, shap_values = shap_engine.get_patient_explanation(patient_index)

# SAFE CONVERSION (THIS FIXES YOUR ERROR)
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
    shap_values = shap_values[:, 1]

# WATERFALL PLOT (SAFE)
fig, ax = plt.subplots()
shap.plots.waterfall(explanation, show=False)
st.pyplot(fig)


# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("⚠️ Key Risk Factors")

shap_df = pd.DataFrame({
    "feature": features,
    "impact": shap_values
}).sort_values("impact", key=lambda x: abs(x), ascending=False)

for _, row in shap_df.head(5).iterrows():
    direction = "increases" if row["impact"] > 0 else "decreases"
    st.write(f"• **{row['feature']}** {direction} risk")


# =========================
# CLINICAL INTERPRETATION
# =========================
st.subheader("🧾 Clinical Interpretation")

if prob > 0.7:
    st.error("High probability of heart disease. Immediate clinical attention required.")
elif prob > 0.4:
    st.warning("Moderate risk detected. Further diagnostic evaluation recommended.")
else:
    st.success("Low risk. Routine monitoring is sufficient.")


# =========================
# PDF REPORT
# =========================
def generate_pdf(index, prob, top_features):
    file_path = f"{OUTPUT_DIR}/patient_{index}_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    story = []

    text = f"""
    <b>Heart Disease Clinical Report</b><br/><br/>
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
# DOWNLOAD REPORT
# =========================
st.subheader("📄 Clinical Report Generator")

if st.button("Generate PDF Report"):
    path = generate_pdf(patient_index, prob, shap_df.head(5))

    with open(path, "rb") as file:
        st.download_button(
            label="⬇️ Download Report",
            data=file,
            file_name=os.path.basename(path),
            mime="application/pdf"
        )


# =========================
# FAIRNESS UI
# =========================
st.subheader("⚖️ Fairness Analysis")

tab1, tab2 = st.tabs(["Gender Fairness", "Age Fairness"])

with tab1:
    f_df, r_gap, a_gap = fairness_engine.evaluate("sex")
    st.dataframe(f_df)
   
    st.success("Gender fairness analysis complete")

with tab2:
    f_df, r_gap, a_gap = fairness_engine.evaluate("age_group")
    st.dataframe(f_df)

    st.success("Age fairness analysis complete")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🧠 Developed by Daniel Godwin | Clinical AI System (Explainable + Fair)")