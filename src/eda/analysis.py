"""
Exploratory Data Analysis utilities.

Responsibility: produce descriptive statistics and plots for a dataset.
This module is deliberately decoupled from loading/validation/preprocessing
used at training/inference time — it is meant to be invoked from notebooks
or a one-off script only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }


def numerical_summary(df: pd.DataFrame, numerical_columns: Sequence[str]) -> pd.DataFrame:
    if not numerical_columns:
        return pd.DataFrame()
    return df[list(numerical_columns)].describe().transpose()


def categorical_summary(df: pd.DataFrame, categorical_columns: Sequence[str]) -> pd.DataFrame:
    if not categorical_columns:
        return pd.DataFrame()
    rows = []
    for col in categorical_columns:
        value_counts = df[col].value_counts(dropna=False)
        rows.append(
            {
                "column": col,
                "n_unique": int(df[col].nunique(dropna=True)),
                "top_value": value_counts.index[0] if len(value_counts) else None,
                "top_frequency": int(value_counts.iloc[0]) if len(value_counts) else 0,
                "n_missing": int(df[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def target_distribution(df: pd.DataFrame, target_column: str) -> pd.Series:
    return df[target_column].value_counts(dropna=False)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df)) * 100
    report = pd.DataFrame({"missing_count": missing, "missing_percentage": pct})
    return report[report["missing_count"] > 0].sort_values(
        "missing_percentage", ascending=False
    )


def plot_numerical_distributions(
    df: pd.DataFrame, numerical_columns: Sequence[str], output_dir: Path, max_cols: int = 20
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for col in list(numerical_columns)[:max_cols]:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].dropna().hist(bins=30, ax=ax)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(output_dir / f"dist_{col}.png")
        plt.close(fig)


def plot_missing_values(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("No missing values found; skipping missing-value plot.")
        return
    fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.3)))
    missing.plot(kind="barh", ax=ax)
    ax.set_xlabel("Fraction missing")
    ax.set_title("Missing values by column")
    fig.tight_layout()
    fig.savefig(output_dir / "missing_values.png")
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame, target_column: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    df[target_column].value_counts(dropna=False).plot(kind="bar", ax=ax)
    ax.set_title(f"Target distribution: {target_column}")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png")
    plt.close(fig)


def generate_eda_report(
    df: pd.DataFrame,
    target_column: str,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    output_dir: Path,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Runs the full EDA suite and writes a JSON summary + optional plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "overview": dataset_overview(df),
        "numerical_summary": numerical_summary(df, numerical_columns).to_dict(),
        "categorical_summary": categorical_summary(df, categorical_columns).to_dict(),
        "target_distribution": target_distribution(df, target_column).to_dict(),
        "missing_values": missing_value_report(df).to_dict(),
    }

    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if make_plots:
        plot_numerical_distributions(df, numerical_columns, output_dir / "plots")
        plot_missing_values(df, output_dir / "plots")
        plot_target_distribution(df, target_column, output_dir / "plots")

    logger.info("EDA report written to %s", output_dir)
    return summary