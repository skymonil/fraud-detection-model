
from pathlib import Path

import pandas as pd
import pytest

from fastapi.testclient import TestClient
from src.api import app


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "test.csv"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_payload():
    df = pd.read_csv(TEST_DATA_PATH)

    # Select a fraud transaction detected by the model
    from src.api import THRESHOLD
    import joblib

    model_path = (
        PROJECT_ROOT
        / "artifacts"
        / "models"
        / "logistic_regression_baseline.joblib"
    )

    model = joblib.load(model_path)

    X = df.drop(columns=["Class"])
    probabilities = model.predict_proba(X)[:, 1]

    detected_indices = df.index[
        (df["Class"] == 1) & (probabilities >= THRESHOLD)
    ]

    assert len(detected_indices) > 0

    index = detected_indices[0]

    return df.drop(columns=["Class"]).iloc[index].to_dict()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["classification"] in ["Fraud", "Legitimate"]
    assert 0 <= result["fraud_probability"] <= 1
    assert result["threshold"] == 0.05


def test_fraud_prediction(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["classification"] == "Fraud"


def test_missing_feature(client, valid_payload):
    payload = valid_payload.copy()
    payload.pop("Amount")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_feature(client, valid_payload):
    payload = valid_payload.copy()
    payload["CustomerID"] = 12345

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_negative_amount(client, valid_payload):
    payload = valid_payload.copy()
    payload["Amount"] = -100

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_input_type(client, valid_payload):
    payload = valid_payload.copy()
    payload["Amount"] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
