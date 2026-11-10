"""
Automated EDA for the ULB Credit Card Fraud dataset.

Usage:
  pip install pandas numpy matplotlib seaborn
  python eda.py
  python eda.py --input data/raw/creditcard.csv --output reports/eda
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/creditcard.csv")
    parser.add_argument("--output", default="reports/eda")
    parser.add_argument("--target", default="Class")
    args = parser.parse_args()

    source = Path(args.input)
    out = Path(args.output)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    df = pd.read_csv(source)
    target = args.target
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Columns: {df.columns.tolist()}")

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    features = [c for c in numeric if c != target]

    # Dataset and data-quality summaries
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(4),
    }).sort_values("missing_count", ascending=False)
    missing.to_csv(out / "missing_values.csv")

    df[numeric].describe().T.to_csv(out / "numeric_summary.csv")
    counts = df[target].value_counts(dropna=False).sort_index()
    distribution = pd.DataFrame({
        "count": counts,
        "percentage": (counts / len(df) * 100).round(4),
    })
    distribution.to_csv(out / "class_distribution.csv")

    duplicate_count = int(df.duplicated().sum())

    # Feature summaries by target class
    if features:
        df.groupby(target)[features].agg(
            ["mean", "median", "std", "min", "max"]
        ).to_csv(out / "class_numeric_summary.csv")

    # Correlation with target
    corr_target = pd.Series(dtype=float)
    if target in numeric and len(numeric) > 1:
        corr_target = (
            df[numeric].corr(numeric_only=True)[target]
            .drop(labels=[target], errors="ignore")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        corr_target.rename("correlation_with_target").to_csv(
            out / "correlation_with_target.csv"
        )

    # Human-readable report
    with (out / "eda_report.txt").open("w", encoding="utf-8") as f:
        f.write("AUTOMATED EDA REPORT\n" + "=" * 65 + "\n")
        f.write(f"File: {source}\nRows: {len(df):,}\nColumns: {len(df.columns)}\n")
        f.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write("\nCOLUMN TYPES\n" + df.dtypes.to_string() + "\n")
        f.write("\nMISSING VALUES\n" + missing.to_string() + "\n")
        f.write(f"\nDuplicate rows: {duplicate_count:,} "
                f"({duplicate_count / len(df) * 100:.4f}%)\n")
        f.write("\nTARGET / CLASS DISTRIBUTION\n" + distribution.to_string() + "\n")
        f.write("\nNUMERIC SUMMARY\n" + df[numeric].describe().T.to_string() + "\n")
        if features:
            grouped = df.groupby(target)[features].agg(
                ["mean", "median", "std", "min", "max"]
            )
            f.write("\nNUMERIC FEATURES BY CLASS\n" + grouped.to_string() + "\n")
        f.write("\nCORRELATION WITH TARGET\n")
        f.write(corr_target.to_string() if not corr_target.empty else "Not available")
        f.write("\n\nINITIAL MODELLING NOTES\n")
        f.write("- Accuracy alone can be misleading when one class is rare.\n")
        f.write("- Evaluate with precision, recall, F1 and PR-AUC, not accuracy alone.\n")
        f.write("- Use a stratified train/test split to preserve class proportions.\n")
        f.write("- Fit preprocessing on training data only to avoid data leakage.\n")
        f.write("- Select the fraud threshold based on missed-fraud vs false-alarm trade-offs.\n")
        if "Time" in df.columns:
            f.write("- In this dataset, Time is elapsed seconds, not a calendar timestamp.\n")
        if all(c in df.columns for c in [f"V{i}" for i in range(1, 29)]):
            f.write("- V1-V28 are anonymized PCA-derived features; business meanings are unavailable.\n")

    # Plots
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=target)
    ax.set_title("Class distribution")
    ax.set_ylabel("Number of transactions")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(plots / "class_distribution.png", dpi=150)
    plt.close()

    if "Amount" in df.columns:
        temp = df[[target, "Amount"]].copy()
        temp["log1p_Amount"] = np.log1p(temp["Amount"].clip(lower=0))
        plt.figure(figsize=(9, 5))
        sns.histplot(data=temp, x="log1p_Amount", hue=target, bins=50,
                     stat="density", common_norm=False, element="step")
        plt.title("Transaction amount by class (log scale)")
        plt.xlabel("log(1 + Amount)")
        plt.tight_layout()
        plt.savefig(plots / "amount_by_class.png", dpi=150)
        plt.close()
        df.groupby(target)["Amount"].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).to_csv(out / "amount_by_class.csv")

    if "Time" in df.columns:
        plt.figure(figsize=(9, 5))
        sns.histplot(data=df, x="Time", hue=target, bins=60, element="step")
        plt.title("Transaction time by class")
        plt.xlabel("Elapsed seconds from first transaction")
        plt.tight_layout()
        plt.savefig(plots / "time_by_class.png", dpi=150)
        plt.close()

    if len(numeric) > 1:
        plt.figure(figsize=(max(10, len(numeric) * 0.45), max(8, len(numeric) * 0.4)))
        sns.heatmap(df[numeric].corr(), cmap="coolwarm", center=0)
        plt.title("Numeric feature correlation matrix")
        plt.tight_layout()
        plt.savefig(plots / "correlation_heatmap.png", dpi=150)
        plt.close()

    if not corr_target.empty:
        top = corr_target.head(15).sort_values()
        plt.figure(figsize=(9, 6))
        sns.barplot(x=top.values, y=top.index, orient="h")
        plt.title("Top 15 features correlated with target")
        plt.xlabel("Correlation with target")
        plt.tight_layout()
        plt.savefig(plots / "top_target_correlations.png", dpi=150)
        plt.close()

    for col in [f"V{i}" for i in [1, 2, 3, 4, 10, 12, 14, 17]]:
        if col in df.columns:
            plt.figure(figsize=(8, 4.5))
            sns.histplot(data=df, x=col, hue=target, bins=60,
                         stat="density", common_norm=False, element="step")
            plt.title(f"{col} distribution by class")
            plt.tight_layout()
            plt.savefig(plots / f"{col.lower()}_by_class.png", dpi=150)
            plt.close()

    (out / "duplicate_summary.txt").write_text(
        f"Rows: {len(df):,}\nDuplicate rows: {duplicate_count:,}\n"
        f"Duplicate percentage: {duplicate_count / len(df) * 100:.4f}%\n",
        encoding="utf-8"
    )

    print("EDA completed.")
    print(f"Read this first: {out / 'eda_report.txt'}")
    print(f"Plots are saved in: {plots}")


if __name__ == "__main__":
    main()
