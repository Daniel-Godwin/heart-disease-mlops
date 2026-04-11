from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    return {"risk_probability": float(prob)}