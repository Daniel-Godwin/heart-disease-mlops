# 🏥 Heart Disease Prediction & Explainable AI System (MLOps Project)

## 📌 Overview

This project presents a **production-ready Machine Learning and Explainable AI (XAI) system** for predicting heart disease risk. It integrates **model training, explainability (SHAP), clinical reporting, and a Streamlit dashboard** within a structured MLOps pipeline.

The system is designed as a **clinical decision support prototype**, enabling interpretable predictions for healthcare applications.

---

## 🚀 Key Features

* ✅ Heart disease prediction using Machine Learning (Random Forest)
* 🧠 Explainable AI using SHAP (global + local explanations)
* 📊 Feature importance and patient-level interpretation
* 🏥 Clinical PDF report generation
* 🎨 Interactive Streamlit dashboard
* ⚡ Modular MLOps architecture
* 🔁 CI/CD pipeline with GitHub Actions
* 📦 Reproducible environment using `requirements.txt`

---

## 🧠 Machine Learning Pipeline

* Data preprocessing and feature selection
* Model training using factory pattern
* Evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
* Model persistence using `joblib`

---

## 🔍 Explainable AI (SHAP)

* Global feature importance (summary & bar plots)
* Local explanations (patient-level SHAP values)
* Risk contribution analysis:

  * Positive SHAP → increases risk
  * Negative SHAP → decreases risk

---

## 🏥 Clinical Decision Support

The system generates **automated medical-style reports** including:

* Patient risk probability
* Top contributing features
* Clinical interpretation (low / moderate / high risk)
* PDF report for real-world usability

---

## 🎨 Streamlit Dashboard

Interactive UI for:

* Selecting patient data
* Viewing predictions
* Visualizing SHAP feature impact
* Generating downloadable clinical reports

Run locally:

```bash
streamlit run app/app.py
```

---

## ⚡ API (FastAPI - Optional Extension)

Provides REST endpoints for prediction:

```bash
POST /predict
```

Returns:

```json
{
  "risk_probability": 0.82
}
```

---

## 🏗️ Project Structure

```
heart_disease_mlops/
│
├── app/                  # Streamlit dashboard
├── api/                  # FastAPI backend
├── src/                  # ML pipeline
├── monitoring/           # SHAP & drift monitoring
├── models/               # Saved models
├── data/                 # Dataset
├── reports/              # Generated reports
├── tests/                # Unit tests
├── docker/               # Docker setup
├── .github/workflows/    # CI/CD pipeline
└── requirements.txt
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-mlops.git
cd heart-disease-mlops
```

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train Model

```bash
python -m src.train_model
```

### Generate SHAP Explanations

```bash
python -m monitoring.explainability
```

### Generate Clinical Report

```bash
python -m src.clinical_shap_report
```

### Run Dashboard

```bash
streamlit run app/app.py
```

---

## 🔁 CI/CD Pipeline

GitHub Actions automatically:

* Installs dependencies
* Runs tests
* Executes training pipeline
* Validates explainability module

---

## 🐳 Docker (Optional)

Build and run:

```bash
docker build -t heart-mlops .
docker run -p 8501:8501 heart-mlops
```

---

## 📊 Technologies Used

* Python
* Scikit-learn
* SHAP
* Pandas & NumPy
* Streamlit
* MLflow
* FastAPI
* ReportLab
* Docker
* GitHub Actions

---

## 🎯 Applications

* Clinical decision support systems
* Explainable AI in healthcare
* Medical risk prediction
* AI-assisted diagnosis tools

---

## 📈 Future Improvements

* Real-time data streaming
* Advanced model tuning (Optuna)
* Bias & fairness analysis
* Multi-disease prediction system
* Cloud deployment (AWS / Azure)

---

## 👨‍💻 Author

**Godwin Daniel**
Master’s Candidate in Computer Science (AI Focus)
Adamawa State, Nigeria

---

## 📜 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgment

This project demonstrates the integration of **Machine Learning, Explainable AI, and MLOps** into a real-world healthcare application.
