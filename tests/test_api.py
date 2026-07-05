"""Integration tests for the FastAPI service (api/main.py)"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PATIENT = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 130, "chol": 230,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers lifespan (artifact loading)
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == 13
    assert "age" in body["features"]


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_level"] in ("low", "medium", "high")


def test_predict_missing_field(client):
    bad = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_out_of_range(client):
    bad = dict(VALID_PATIENT, age=500)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_wrong_type(client):
    bad = dict(VALID_PATIENT, chol="not-a-number")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
