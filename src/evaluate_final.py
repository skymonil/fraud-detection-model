
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "artifacts/models/logistic_regression_baseline.joblib"
TEST_PATH = PROJECT_ROOT / "data/processed/test.csv"
REPORT_DIR = PROJECT_ROOT / "reports/final"

TARGET = "Class"
THRESHOLD = 0.05


def main():
    print("Loading selected model and test dataset...")

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    print(f"Test dataset: {X_test.shape}")
    print(f"Fraud cases: {y_test.sum()}")

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Apply selected validation threshold
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # Calculate metrics
    metrics = {
        "threshold": THRESHOLD,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_prob)
    }

    cm = confusion_matrix(y_test, y_pred)

    print("\n========== FINAL TEST RESULTS ==========")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_json(
        REPORT_DIR / "metrics.json",
        orient="records",
        indent=4
    )

    pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
