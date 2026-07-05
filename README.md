![CI](https://github.com/Daniel-Godwin/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)
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

## ⚡ API (FastAPI)

Production REST API with input validation, health checks, and prediction logging:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Liveness/readiness check |
| `/model-info` | GET | Model metadata & feature list |
| `/predict` | POST | Patient risk prediction |
| `/docs` | GET | Interactive Swagger UI |

Example request:

```json
POST /predict
{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 130, "chol": 230,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
}
```

Response:

```json
{
  "prediction": 1,
  "risk_probability": 0.7607,
  "risk_level": "high",
  "model_version": "1.0.0"
}
```

---

## 📉 Drift Detection & Monitoring

Data drift is detected with **Kolmogorov–Smirnov tests** and **Population Stability Index (PSI)** per feature:

```bash
python run.py --stage drift
```

Produces `reports/drift_report.json` with per-feature statistics and a retraining recommendation. Thresholds: KS p-value < 0.05, PSI > 0.2, overall drift if ≥30% of features shift. All API predictions are logged as structured JSON lines (`logs/predictions.jsonl`) for downstream monitoring.

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

### Full Pipeline (Orchestrated)

```bash
python run.py                    # preprocess -> train -> drift check
python run.py --stage preprocess # single stage
python run.py --stage train
python run.py --stage drift
```

### Run Tests

```bash
pytest tests/ -v
```

29 tests cover preprocessing, model artifacts, inference, the API (integration tests), and a **model quality gate** (accuracy ≥ 0.75, ROC-AUC ≥ 0.80 on held-out data).

### Generate SHAP Explanations

```bash
python -m monitoring.explainability
```

### Run Dashboard

```bash
streamlit run app/app.py
```

---

## 🔁 CI/CD Pipeline

GitHub Actions (on every push/PR to `main`):

1. Install dependencies (with pip caching)
2. Run full test suite — **build fails if any test fails**
3. Run drift detection and upload the drift report as a CI artifact
4. Build the Docker image (only if tests pass)

---

## 🐳 Docker

Single container (API):

```bash
docker build -f docker/Dockerfile -t heart-mlops .
docker run -p 8000:8000 heart-mlops
```

Full stack (API + Dashboard + MLflow UI):

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Service | Port |
|---------|------|
| FastAPI | 8000 |
| Streamlit Dashboard | 8501 |
| MLflow UI | 5000 |

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
