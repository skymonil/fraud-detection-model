
from pathlib import Path

import joblib
import pandas as pd
import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "preprocessed" / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "logistic_regression_baseline.joblib"
)

API_URL = "http://127.0.0.1:8000/predict"

THRESHOLD = 0.05


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

df = pd.read_csv(TEST_DATA_PATH)

# Select one actual transaction
model = joblib.load(MODEL_PATH)

fraud_transactions = df[df["Class"] == 1].copy()

fraud_probabilities = model.predict_proba(
    fraud_transactions.drop(columns=["Class"])
)[:, 1]

detected_indices = fraud_transactions.index[
    fraud_probabilities >= THRESHOLD
]

if len(detected_indices) == 0:
    raise RuntimeError("No correctly detected fraud transactions found")

selected_index = detected_indices[0]
transaction = df.loc[[selected_index]]

# Separate input features and actual target
X = transaction.drop(columns=["Class"])
actual_label = int(transaction["Class"].iloc[0])

print("\n========== TRANSACTION DETAILS ==========")
print(f"Dataset row: {selected_index}")
print(f"Actual Class: {actual_label}")


# --------------------------------------------------
# Local Model Prediction
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

local_probability = float(
    model.predict_proba(X)[0][1]
)

local_prediction = int(
    local_probability >= THRESHOLD
)

print("\n========== LOCAL MODEL PREDICTION ==========")
print(f"Fraud Probability: {local_probability:.6f}")
print(f"Prediction: {local_prediction}")


# --------------------------------------------------
# API Prediction
# --------------------------------------------------

payload = X.iloc[0].to_dict()

response = requests.post(
    API_URL,
    json=payload,
    timeout=10
)

response.raise_for_status()

api_result = response.json()

api_prediction = api_result["prediction"]
api_probability = api_result["fraud_probability"]

print("\n========== API PREDICTION ==========")
print(f"Fraud Probability: {api_probability:.6f}")
print(f"Prediction: {api_prediction}")
print(f"Classification: {api_result['classification']}")


# --------------------------------------------------
# Compare Results
# --------------------------------------------------

print("\n========== PREDICTION COMPARISON ==========")

print(f"Actual Label:       {actual_label}")
print(f"Local Prediction:   {local_prediction}")
print(f"API Prediction:     {api_prediction}")

print(
    f"Probability Match:  "
    f"{abs(local_probability - api_probability) < 1e-5}"
)

print(
    f"Prediction Match:   "
    f"{local_prediction == api_prediction}"
)

print(
    f"Actual Label Match: "
    f"{local_prediction == actual_label}"
)
