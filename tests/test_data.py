
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"

DATASETS = ["train", "validation", "test"]


@pytest.fixture(params=DATASETS)
def dataset(request):
    """Load each preprocessed dataset for validation."""
    file_path = DATA_DIR / f"{request.param}.csv"

    assert file_path.exists(), f"Dataset not found: {file_path}"

    return request.param, pd.read_csv(file_path)


def test_dataset_not_empty(dataset):
    name, df = dataset

    assert not df.empty, f"{name} dataset is empty"


def test_expected_columns(dataset):
    name, df = dataset

    assert len(df.columns) == 31, (
        f"{name} dataset should contain 31 columns"
    )

    assert "Class" in df.columns


def test_no_missing_values(dataset):
    name, df = dataset

    assert not df.isnull().any().any(), (
        f"{name} dataset contains missing values"
    )


def test_valid_target_labels(dataset):
    name, df = dataset

    assert set(df["Class"].unique()).issubset({0, 1}), (
        f"{name} contains invalid target labels"
    )


def test_both_classes_present(dataset):
    name, df = dataset

    assert set(df["Class"].unique()) == {0, 1}, (
        f"{name} must contain both legitimate and fraud transactions"
    )


def test_features_are_finite(dataset):
    name, df = dataset

    features = df.drop(columns=["Class"])

    assert np.isfinite(features.to_numpy()).all(), (
        f"{name} contains infinite or non-finite feature values"
    )
