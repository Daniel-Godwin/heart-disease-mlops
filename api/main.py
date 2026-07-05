"""
main.py
-------
Production FastAPI service for heart disease risk prediction.

Endpoints:
    GET  /            -> service info
    GET  /health      -> liveness/readiness check
    GET  /model-info  -> model metadata
    POST /predict     -> single patient prediction

Fixes a critical bug in the original version: the model was trained on
StandardScaler-transformed features, so raw inputs are now scaled with
the persisted scaler before prediction.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils import load_model, load_features, load_scaler, prepare_input, get_risk_level
from monitoring.logging import log_prediction

APP_VERSION = "1.0.0"

# Loaded once at startup, not per-request
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()
    state["features"] = load_features()
    state["scaler"] = load_scaler()
    yield
    state.clear()


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Clinical decision support: heart disease risk prediction with explainable ML.",
    version=APP_VERSION,
    lifespan=lifespan,
)


class PatientData(BaseModel):
    """Raw (unscaled) clinical measurements for one patient."""
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression (exercise vs rest)")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 130, "chol": 230,
                "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
                "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    risk_probability: float
    risk_level: str
    model_version: str


@app.get("/")
def root():
    return {
        "service": "Heart Disease Prediction API",
        "version": APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    ready = all(k in state for k in ("model", "features", "scaler"))
    if not ready:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/model-info")
def model_info():
    model = state["model"]
    return {
        "model_type": type(model).__name__,
        "n_features": len(state["features"]),
        "features": state["features"],
        "version": APP_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    try:
        X = prepare_input(
            patient.model_dump(),
            feature_names=state["features"],
            scaler=state["scaler"],
        )
        model = state["model"]
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        risk_level = get_risk_level(probability)

        log_prediction(patient.model_dump(), prediction, probability,
                       risk_level, model_version=APP_VERSION)

        return PredictionResponse(
            prediction=prediction,
            risk_probability=round(probability, 4),
            risk_level=risk_level,
            model_version=APP_VERSION,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
