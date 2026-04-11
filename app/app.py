import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# PAGE CONFIG (MUST BE FIRST)
# =========================
st.set_page_config(page_title="Clinical SHAP Dashboard", layout="wide")


# =========================
# HEADER
# =========================
st.markdown("""
# 🏥 Heart Disease Clinical Decision Support System

This system uses Machine Learning and Explainable AI (SHAP) to predict heart disease risk and provide interpretable insights for clinical decision-making.
""")


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
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

df = pd.read_csv(DATA_PATH)
X = df[features]

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 55, 100],
    labels=["Young", "Middle", "Old"]
)


# =========================
# SHAP EXPLAINER
# =========================
@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(model)

explainer = load_explainer()
shap_exp = explainer(X)


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Patient Selection")

patient_index = st.sidebar.slider(
    "Select patient index",
    0,
    len(X) - 1,
    0
)

patient_data = X.iloc[patient_index]
patient_df = pd.DataFrame([patient_data], columns=features)


# =========================
# PREDICTION
# =========================
prob = model.predict_proba(patient_df)[0][1]

if prob > 0.7:
    prediction = "HIGH RISK"
elif prob > 0.4:
    prediction = "MODERATE RISK"
else:
    prediction = "LOW RISK"


# =========================
# DISPLAY RESULTS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Prediction Result")

    if prob > 0.7:
        st.error(f"🔴 HIGH RISK ({prob:.2f})")
    elif prob > 0.4:
        st.warning(f"🟡 MODERATE RISK ({prob:.2f})")
    else:
        st.success(f"🟢 LOW RISK ({prob:.2f})")

with col2:
    st.subheader("📊 Patient Data")
    st.dataframe(patient_df)


# =========================
# SHAP VALUES
# =========================
shap_values = shap_exp.values[patient_index, :, 1]

shap_df = pd.DataFrame({
    "feature": features,
    "impact": shap_values
})

shap_df = shap_df.loc[
    shap_df["impact"].abs().sort_values(ascending=False).index
]

top_features = shap_df.head(5)


# =========================
# SHAP BAR CHART
# =========================
st.subheader("🔍 SHAP Feature Impact")
st.bar_chart(shap_df.set_index("feature"))


# =========================
# SHAP WATERFALL
# =========================
st.subheader("📉 Detailed SHAP Explanation")

explanation = shap.Explanation(
    values=shap_values,
    base_values=shap_exp.base_values[patient_index, 1],
    data=patient_data,
    feature_names=features
)

fig, ax = plt.subplots()
shap.plots.waterfall(explanation, show=False)
st.pyplot(fig)


# =========================
# TOP FEATURES
# =========================
st.subheader("⚠️ Top Contributing Factors")

for _, row in top_features.iterrows():
    direction = "increases" if row["impact"] > 0 else "decreases"
    st.write(f"• **{row['feature']}** {direction} risk")


# =========================
# CLINICAL INTERPRETATION
# =========================
st.subheader("🧠 AI Clinical Interpretation")

if prob > 0.7:
    st.write("The patient shows a high likelihood of heart disease. Immediate clinical evaluation is strongly recommended.")
elif prob > 0.4:
    st.write("The patient has a moderate risk. Further diagnostic tests are advised.")
else:
    st.write("The patient is at low risk. Routine monitoring is sufficient.")


# =========================
# PDF GENERATION
# =========================
def generate_pdf(index, prob, top_features):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# DOWNLOAD REPORT
# =========================
st.subheader("📄 Generate Clinical Report")

if st.button("Generate PDF Report"):
    path = generate_pdf(patient_index, prob, top_features)

    with open(path, "rb") as file:
        st.download_button(
            label="⬇️ Download Report",
            data=file,
            file_name=os.path.basename(path),
            mime="application/pdf"
        )


# =========================
# FAIRNESS ANALYSIS
# =========================
def compute_fairness(df, model, feature):

    X_full = df[features]
    y_full = df["target"]

    groups = df[feature].unique()
    results = []

    for g in groups:
        idx = df[feature] == g

        X_group = X_full[idx]
        y_group = y_full[idx]

        if len(y_group) < 10:
            continue

        y_pred = model.predict(X_group)

        acc = (y_pred == y_group).mean()
        rec = ((y_pred == 1) & (y_group == 1)).sum() / max((y_group == 1).sum(), 1)

        results.append({
            "group": str(g),
            "size": len(y_group),
            "accuracy": acc,
            "recall": rec
        })

    fairness_df = pd.DataFrame(results)

    recall_gap = fairness_df["recall"].max() - fairness_df["recall"].min()
    acc_gap = fairness_df["accuracy"].max() - fairness_df["accuracy"].min()

    return fairness_df, recall_gap, acc_gap


# =========================
# FAIRNESS DASHBOARD
# =========================
st.subheader("⚖️ Fairness Analysis")

tab1, tab2 = st.tabs(["Gender Bias", "Age Bias"])

# -------------------------
# GENDER FAIRNESS
# -------------------------
with tab1:
    fairness_df, recall_gap, acc_gap = compute_fairness(df, model, "sex")

    st.write("### Group Performance (Sex)")
    st.dataframe(fairness_df)

    st.write(f"📉 Recall Gap: {recall_gap:.3f}")
    st.write(f"📉 Accuracy Gap: {acc_gap:.3f}")

    if recall_gap > 0.1:
        st.error("⚠️ Potential gender bias detected")
    else:
        st.success("✅ Model appears fair across gender")


# -------------------------
# AGE FAIRNESS
# -------------------------
with tab2:
    fairness_df, recall_gap, acc_gap = compute_fairness(df, model, "age_group")

    st.write("### Group Performance (Age)")
    st.dataframe(fairness_df)

    st.write(f"📉 Recall Gap: {recall_gap:.3f}")
    st.write(f"📉 Accuracy Gap: {acc_gap:.3f}")

    if recall_gap > 0.1:
        st.error("⚠️ Potential age bias detected")
    else:
        st.success("✅ Model appears fair across age groups")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Developed by **Godwin Daniel** | Explainable AI in Healthcare")