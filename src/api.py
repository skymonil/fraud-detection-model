
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "logistic_regression_baseline.joblib"
)

THRESHOLD = 0.05

FEATURE_NAMES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class Transaction(BaseModel):
    """
    Input schema for a single credit card transaction.
    """

    model_config = ConfigDict(extra="forbid")

    Time: Annotated[float, Field(ge=0)]
    
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: Annotated[float, Field(ge=0)]


# --------------------------------------------------
# Response Schema
# --------------------------------------------------

class PredictionResponse(BaseModel):
    prediction: int
    classification: str
    fraud_probability: float
    threshold: float


# --------------------------------------------------
# Application Lifecycle
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model once when the API starts.
    """

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    app.state.model = joblib.load(MODEL_PATH)

    expected_features = list(
        app.state.model.feature_names_in_
    )

    if expected_features != FEATURE_NAMES:
        raise RuntimeError(
            "Model feature names do not match API input schema"
        )

    print("Fraud detection model loaded successfully.")

    yield

    # Cleanup when application shuts down
    del app.state.model


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Credit card fraud detection using Machine Learning",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/health")
def health(request: Request):

    model_loaded = hasattr(request.app.state, "model")

    return {
        "status": "healthy" if model_loaded else "unavailable",
        "model_loaded": model_loaded,
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    transaction: Transaction,
    request: Request
):

    model = request.app.state.model

    try:
        # Convert request into DataFrame
        transaction_data = pd.DataFrame(
            [transaction.model_dump()],
            columns=FEATURE_NAMES,
        )

        # Get fraud probability
        fraud_probability = float(
            model.predict_proba(transaction_data)[0][1]
        )

        # Apply selected decision threshold
        prediction = int(
            fraud_probability >= THRESHOLD
        )

        classification = (
            "Fraud" if prediction == 1 else "Legitimate"
        )

        return PredictionResponse(
            prediction=prediction,
            classification=classification,
            fraud_probability=round(fraud_probability, 6),
            threshold=THRESHOLD,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc
