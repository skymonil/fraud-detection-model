
from pathlib import Path
import json

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from mlflow_utils import log_training_run
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "smote"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load original datasets
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

# Separate features and target
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_val = val_df.drop(columns=["Class"])
y_val = val_df["Class"]

print("Original training dataset:", X_train.shape)
print("Validation dataset:", X_val.shape)

print("\nOriginal training class distribution:")
print(y_train.value_counts())

print("\nValidation fraud cases:", y_val.sum())

# Build pipeline
# Scaling happens before SMOTE because SMOTE uses distances.
# SMOTE runs only during model.fit(), on training data.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(
        sampling_strategy="auto",
        random_state=42,
        k_neighbors=5
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train model
print("\nTraining Logistic Regression with SMOTE...")

model.fit(X_train, y_train)

print("Training completed!")

# Display the resampled training distribution
resampled_counts = model.named_steps["smote"].fit_resample(
    model.named_steps["scaler"].transform(X_train),
    y_train
)[1].value_counts()

print("\nResampled training class distribution:")
print(resampled_counts)

# Validation predictions - validation data is NOT resampled
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Calculate metrics
metrics = {
    "accuracy": float(accuracy_score(y_val, y_pred)),
    "precision": float(precision_score(y_val, y_pred, zero_division=0)),
    "recall": float(recall_score(y_val, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
    "pr_auc": float(average_precision_score(y_val, y_prob)),
}

cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

print("\n========== SMOTE RESULTS ==========")

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

# Save trained pipeline
model_path = MODEL_DIR / "logistic_regression_smote.joblib"
joblib.dump(model, model_path)

# Save metrics
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
    run_name="smote-logistic-regression",
    params={
        "algorithm": "LogisticRegression",
        "scaler": "StandardScaler",
        "sampling_strategy": "auto",
        "smote_k_neighbors": 5,
        "smote_random_state": 42,
        "max_iter": 1000,
        "random_state": 42,
        "threshold": 0.5,
        "dataset": "deduplicated-creditcard",
    },
    metrics=metrics,
    report_dir=REPORT_DIR,
)