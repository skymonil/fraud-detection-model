
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "logistic_regression_baseline.joblib"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "test.csv"

THRESHOLD = 0.05


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), "Trained model not found"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_data():
    df = pd.read_csv(TEST_DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    return X, y


def test_model_loads(model):
    assert model is not None


def test_model_has_expected_features(model, test_data):
    X, _ = test_data

    assert list(model.feature_names_in_) == list(X.columns)


def test_model_prediction_shape(model, test_data):
    X, _ = test_data

    predictions = model.predict(X)

    assert predictions.shape == (len(X),)


def test_model_probability_shape(model, test_data):
    X, _ = test_data

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)


def test_model_probabilities_are_valid(model, test_data):
    X, _ = test_data

    probabilities = model.predict_proba(X)[:, 1]

    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_threshold_predictions_are_binary(model, test_data):
    X, _ = test_data

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    assert set(np.unique(predictions)).issubset({0, 1})


def test_threshold_detects_fraud(model, test_data):
    X, y = test_data

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    actual_fraud_count = int(y.sum())
    detected_fraud_count = int(((predictions == 1) & (y == 1)).sum())

    assert actual_fraud_count > 0
    assert detected_fraud_count > 0
