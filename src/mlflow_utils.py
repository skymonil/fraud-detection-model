
import os
from pathlib import Path

import mlflow
import mlflow.sklearn


def log_training_run(
    model,
    run_name,
    params,
    metrics,
    report_dir,
):
    """
    Log one model training experiment to the remote MLflow server.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI","https://mlflow.615915.xyz")

    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not configured."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name=run_name) as run:

        # Log model configuration
        mlflow.log_params(params)

        # Log validation metrics
        mlflow.log_metrics(metrics)

        # Log evaluation reports as artifacts
        report_path = Path(report_dir)

        if report_path.exists():
            mlflow.log_artifacts(
                str(report_path),
                artifact_path="reports",
            )

        # Log trained sklearn pipeline
        mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    skops_trusted_types=[
        "imblearn.over_sampling._smote.base.SMOTE",
        "imblearn.pipeline.Pipeline",
    ],
)

        print("\n========== MLFLOW LOGGING ==========")
        print(f"Experiment: fraud-detection")
        print(f"Run name: {run_name}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Tracking URI: {tracking_uri}")
        print("Parameters, metrics, reports and model logged.")
