
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
}

TARGET = "Class"


def load_datasets():
    datasets = {}

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        datasets[name] = pd.read_csv(path)

    return datasets


def get_hashes(df, columns):
    """Generate stable hashes for selected columns."""
    return pd.util.hash_pandas_object(
        df[columns],
        index=False
    )


def audit_within_split(datasets, feature_columns):
    print("\n========== WITHIN-SPLIT DUPLICATES ==========")

    for name, df in datasets.items():
        full_duplicates = df.duplicated().sum()

        feature_duplicates = df.duplicated(
            subset=feature_columns
        ).sum()

        # Same features but different target labels
        label_counts = df.groupby(feature_columns)[TARGET].nunique()
        conflicting_groups = (label_counts > 1).sum()

        print(f"\n{name.upper()}")
        print(f"Total rows: {len(df)}")
        print(f"Exact duplicate rows: {full_duplicates}")
        print(f"Duplicate feature rows: {feature_duplicates}")
        print(f"Feature groups with conflicting labels: {conflicting_groups}")


def audit_between_splits(datasets, feature_columns):
    print("\n========== CROSS-SPLIT OVERLAP ==========")

    split_names = list(datasets.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            name_a = split_names[i]
            name_b = split_names[j]

            df_a = datasets[name_a]
            df_b = datasets[name_b]

            full_hash_a = set(get_hashes(df_a, df_a.columns))
            full_hash_b = set(get_hashes(df_b, df_b.columns))

            feature_hash_a = get_hashes(df_a, feature_columns)
            feature_hash_b = get_hashes(df_b, feature_columns)

            full_overlap = full_hash_a & full_hash_b

            feature_set_a = set(feature_hash_a)
            feature_set_b = set(feature_hash_b)

            feature_overlap = feature_set_a & feature_set_b

            # Map each feature hash to the labels it has in each split
            labels_a = {}
            labels_b = {}

            for feature_hash, label in zip(feature_hash_a, df_a[TARGET]):
                labels_a.setdefault(feature_hash, set()).add(label)

            for feature_hash, label in zip(feature_hash_b, df_b[TARGET]):
                labels_b.setdefault(feature_hash, set()).add(label)

            conflicting = [
                h for h in feature_overlap
                if labels_a[h] != labels_b[h]
            ]

            print(f"\n{name_a.upper()} vs {name_b.upper()}")
            print(f"Shared exact rows (including Class): {len(full_overlap)}")
            print(f"Shared feature vectors: {len(feature_overlap)}")
            print(f"Shared feature vectors with conflicting labels: {len(conflicting)}")


def main():
    datasets = load_datasets()

    feature_columns = [
        col for col in datasets["train"].columns
        if col != TARGET
    ]

    print("========== DATASET SUMMARY ==========")

    for name, df in datasets.items():
        print(f"{name}: {df.shape}")

    audit_within_split(datasets, feature_columns)
    audit_between_splits(datasets, feature_columns)

    print("\n========== AUDIT COMPLETE ==========")
    print("No dataset files were modified.")


if __name__ == "__main__":
    main()
