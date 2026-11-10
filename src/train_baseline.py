
from pathlib import Path
import json
from mlflow_utils import log_training_run

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "baseline"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

# Separate features and target
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_val = val_df.drop(columns=["Class"])
y_val = val_df["Class"]

print("Training dataset:", X_train.shape)
print("Validation dataset:", X_val.shape)

print("\nTraining fraud cases:", y_train.sum())
print("Validation fraud cases:", y_val.sum())

# Build baseline model
# No class weighting or resampling at this stage
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train
print("\nTraining Logistic Regression baseline...")

model.fit(X_train, y_train)

print("Training completed!")

# Validation predictions
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Evaluation metrics
metrics = {
    "accuracy": float(accuracy_score(y_val, y_pred)),
    "precision": float(precision_score(y_val, y_pred, zero_division=0)),
    "recall": float(recall_score(y_val, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
    "pr_auc": float(average_precision_score(y_val, y_prob)),
}

# Confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

print("\n========== BASELINE RESULTS ==========")

for metric, value in metrics.items():
    print(f"{metric}: {value:.6f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

# Save model
model_path = MODEL_DIR / "logistic_regression_baseline.joblib"
joblib.dump(model, model_path)

# Save evaluation metrics
metrics_path = REPORT_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# Save confusion matrix
pd.DataFrame(
    cm,
    index=["Actual_Legitimate", "Actual_Fraud"],
    columns=["Predicted_Legitimate", "Predicted_Fraud"]
).to_csv(REPORT_DIR / "confusion_matrix.csv")

print("\nArtifacts saved:")
print("Model:", model_path)
print("Metrics:", metrics_path)
print("Confusion matrix:", REPORT_DIR / "confusion_matrix.csv")

log_training_run(
    model=model,
    run_name="baseline-logistic-regression",
    params={
        "algorithm": "LogisticRegression",
        "scaler": "StandardScaler",
        "class_weight": "None",
        "max_iter": 1000,
        "random_state": 42,
        "threshold": 0.5,
        "dataset": "deduplicated-creditcard",
    },
    metrics=metrics,
    report_dir=REPORT_DIR,
)