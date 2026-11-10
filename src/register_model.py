
import os
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.615915.xyz")

if not TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI is not configured.")

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

experiment = client.get_experiment_by_name("fraud-detection")

if experiment is None:
    raise RuntimeError("Experiment not found.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.run_name = 'baseline-logistic-regression'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError("Baseline run not found.")

run = runs[0]

model_uri = f"runs:/{run.info.run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name="fraud-detection-model",
)

print("Model registered successfully!")
print("Model name:", result.name)
print("Model version:", result.version)
print("Run ID:", run.info.run_id)
