
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"


@pytest.fixture(scope="module")
def datasets():
    return {
        "train": pd.read_csv(DATA_DIR / "train.csv"),
        "validation": pd.read_csv(DATA_DIR / "validation.csv"),
        "test": pd.read_csv(DATA_DIR / "test.csv"),
    }


def test_split_proportions(datasets):
    train = len(datasets["train"])
    validation = len(datasets["validation"])
    test = len(datasets["test"])

    total = train + validation + test

    assert abs(train / total - 0.70) < 0.01
    assert abs(validation / total - 0.15) < 0.01
    assert abs(test / total - 0.15) < 0.01


def test_no_duplicate_transactions_across_splits(datasets):
    train = datasets["train"].drop(columns=["Class"])
    validation = datasets["validation"].drop(columns=["Class"])
    test = datasets["test"].drop(columns=["Class"])

    train_set = set(map(tuple, train.to_numpy()))
    validation_set = set(map(tuple, validation.to_numpy()))
    test_set = set(map(tuple, test.to_numpy()))

    assert train_set.isdisjoint(validation_set)
    assert train_set.isdisjoint(test_set)
    assert validation_set.isdisjoint(test_set)


def test_fraud_distribution_preserved(datasets):
    for name, df in datasets.items():
        fraud_ratio = df["Class"].mean()

        assert 0 < fraud_ratio < 1, (
            f"{name} has an invalid fraud distribution"
        )


def test_all_splits_have_same_features(datasets):
    train_columns = set(datasets["train"].columns)

    for name, df in datasets.items():
        assert set(df.columns) == train_columns, (
            f"{name} has different columns"
        )
