
from pathlib import Path
import json

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "class_weighted"

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

# Calculate class weights for visibility
class_counts = y_train.value_counts()
total_samples = len(y_train)
number_of_classes = y_train.nunique()

class_weights = {
    int(cls): total_samples / (number_of_classes * count)
    for cls, count in class_counts.items()
}

print("\nCalculated class weights:", class_weights)

# Build class-weighted Logistic Regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train
print("\nTraining class-weighted Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed!")

# Validation predictions
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

print("\n========== CLASS-WEIGHTED RESULTS ==========")

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
model_path = MODEL_DIR / "logistic_regression_weighted.joblib"
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
