
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "Class"
RANDOM_STATE = 42


def main():
    # Load original dataset
    df = pd.read_csv(RAW_DATA)

    print("========== ORIGINAL DATASET ==========")
    print(f"Original rows: {len(df)}")
    print(f"Original fraud cases: {df[TARGET].sum()}")

    # Remove exact duplicate records
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    print("\n========== DEDUPLICATION ==========")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Rows after deduplication: {len(df)}")
    print(f"Fraud cases after deduplication: {df[TARGET].sum()}")

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # First split: 70% training, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Second split: divide temporary data equally into validation and test
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # Recombine features and target
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    validation_df = X_validation.copy()
    validation_df[TARGET] = y_validation

    test_df = X_test.copy()
    test_df[TARGET] = y_test

    # Save processed datasets
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    validation_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    # Display split summary
    print("\n========== SPLIT SUMMARY ==========")

    for name, dataset in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Testing", test_df)
    ]:
        print(f"\n{name} dataset: {dataset.shape}")
        print(f"Legitimate: {(dataset[TARGET] == 0).sum()}")
        print(f"Fraud: {(dataset[TARGET] == 1).sum()}")

    print("\nProcessed datasets saved successfully.")


if __name__ == "__main__":
    main()
