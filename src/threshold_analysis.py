
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "threshold_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load validation data
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_val = val_df.drop(columns=["Class"])
y_val = val_df["Class"]

# Load trained models
models = {
    "baseline": joblib.load(
        "artifacts/models/logistic_regression_baseline.joblib"
    ),
    "class_weighted": joblib.load(
        "artifacts/models/logistic_regression_weighted.joblib"
    ),
    "smote": joblib.load(
        "artifacts/models/logistic_regression_smote.joblib"
    ),
}

# Thresholds to evaluate
thresholds = [round(x, 2) for x in
              [0.05, 0.10, 0.15, 0.20, 0.25,
               0.30, 0.35, 0.40, 0.45, 0.50,
               0.55, 0.60, 0.65, 0.70, 0.75,
               0.80, 0.85, 0.90, 0.95]]

all_results = []

print("Starting threshold analysis...")

for model_name, model in models.items():

    print(f"\nAnalyzing model: {model_name}")

    # Generate probabilities once per model
    probabilities = model.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(y_val, probabilities)

    for threshold in thresholds:

        # Convert probabilities into predictions
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        all_results.append({
            "model": model_name,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "pr_auc": pr_auc,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "fraud_flagged": int(tp + fp),
        })


# Convert results into DataFrame
results_df = pd.DataFrame(all_results)

# Save all threshold results
results_path = OUTPUT_DIR / "threshold_results.csv"
results_df.to_csv(results_path, index=False)

# Find best F1 threshold for each model
best_results = (
    results_df.loc[
        results_df.groupby("model")["f1_score"].idxmax()
    ]
    .reset_index(drop=True)
)

print("\n========== BEST F1 THRESHOLDS ==========")

print(
    best_results[
        ["model", "threshold", "precision", "recall", "f1_score",
         "true_positives", "false_positives", "false_negatives"]
    ].to_string(index=False)
)

best_results.to_csv(
    OUTPUT_DIR / "best_thresholds.csv",
    index=False
)

# Plot Precision, Recall and F1 against threshold
for model_name in models:

    model_results = results_df[
        results_df["model"] == model_name
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        model_results["threshold"],
        model_results["precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        model_results["threshold"],
        model_results["recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        model_results["threshold"],
        model_results["f1_score"],
        marker="o",
        label="F1-score",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Analysis - {model_name}")
    plt.xticks(thresholds, rotation=45)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{model_name}_threshold_plot.png",
        dpi=150,
    )

    plt.close()

print("\nThreshold analysis completed!")
print("Results:", results_path)
print("Best thresholds:", OUTPUT_DIR / "best_thresholds.csv")
print("Plots:", OUTPUT_DIR)
