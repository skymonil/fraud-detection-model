
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_PATH = (
    PROJECT_ROOT / "reports" / "final" / "metrics.json"
)

MINIMUM_METRICS = {
    "precision": 0.70,
    "recall": 0.75,
    "f1_score": 0.75,
    "pr_auc": 0.70,
}


@pytest.fixture(scope="module")
def metrics():
    assert METRICS_PATH.exists(), "Final metrics file not found"

    with open(METRICS_PATH, "r") as file:
        data = json.load(file)

    # The metrics file contains a list with one metrics dictionary
    if isinstance(data, list):
        assert len(data) == 1, "Expected exactly one metrics record"
        data = data[0]

    return data


@pytest.mark.parametrize(
    "metric,minimum",
    MINIMUM_METRICS.items()
)
def test_model_quality(metrics, metric, minimum):
    actual = metrics[metric]

    assert actual >= minimum, (
        f"{metric} is below minimum requirement. "
        f"Expected >= {minimum}, got {actual}"
    )
