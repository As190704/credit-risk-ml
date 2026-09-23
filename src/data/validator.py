"""
Data validation utilities.

Responsibility: check structural/quality properties of a raw dataset and
produce an actionable, structured report.

Design decision — fatal vs. non-fatal checks
---------------------------------------------
- Structural impossibilities (empty dataset, missing target column, target
  entirely NaN) raise `DataValidationError` immediately: there is nothing
  meaningful to report otherwise.
- Content-quality issues (missing values, duplicates, out-of-whitelist
  target values) are collected into a `DataQualityReport` instead of
  crashing, so a caller gets one complete diagnostic pass.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised for fatal, unrecoverable data problems."""


@dataclass
class DataQualityReport:
    n_rows: int
    n_columns: int
    duplicate_rows: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    numerical_columns: List[str]
    categorical_columns: List[str]
    target_distribution: Optional[Dict[Any, int]]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_values": self.missing_values,
            "missing_percentage": self.missing_percentage,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "target_distribution": self.target_distribution,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def has_errors(self) -> bool:
        return len(self.errors) > 0


class DataValidator:
    """Runs a sequence of validation checks against a raw DataFrame."""

    def __init__(
        self,
        target_column: str,
        valid_target_values: Optional[Sequence[Any]] = None,
    ):
        self.target_column = target_column
        self.valid_target_values = valid_target_values

    # ---- fatal checks (raise) ----------------------------------------

    def validate_not_empty(self, df: pd.DataFrame) -> None:
        if df is None or df.shape[0] == 0:
            raise DataValidationError("Dataset is empty (zero rows).")
        if df.shape[1] == 0:
            raise DataValidationError("Dataset has zero columns.")

    def validate_target_exists(self, df: pd.DataFrame) -> None:
        if self.target_column not in df.columns:
            raise DataValidationError(
                f"Required target column '{self.target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    def validate_target_values(self, df: pd.DataFrame) -> List[str]:
        """Returns warnings. Raises DataValidationError for hard failures
        (target entirely missing, or values outside a configured whitelist).
        """
        warnings_found: List[str] = []
        target = df[self.target_column]

        n_missing_target = int(target.isna().sum())
        if n_missing_target == len(target):
            raise DataValidationError("Target column contains only missing values.")
        if n_missing_target > 0:
            warnings_found.append(
                f"Target column has {n_missing_target} missing values "
                f"({n_missing_target / len(target):.2%}); these rows cannot "
                "be used for supervised training."
            )

        unique_values = set(target.dropna().unique())

        if self.valid_target_values is not None:
            allowed = set(self.valid_target_values)
            invalid = unique_values - allowed
            if invalid:
                raise DataValidationError(
                    f"Target column contains values outside the configured "
                    f"valid set {allowed}: found {invalid}"
                )
        else:
            if len(unique_values) != 2:
                warnings_found.append(
                    f"Target column has {len(unique_values)} unique values "
                    f"({sorted(unique_values, key=str)[:10]}); expected a "
                    "binary target. Set `valid_target_values` explicitly if "
                    "this is intentional."
                )
        return warnings_found

    # ---- non-fatal checks (collected into the report) -----------------

    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> int:
        return int(df.duplicated().sum())

    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
        return {k: int(v) for k, v in df.isna().sum().to_dict().items()}

    @staticmethod
    def infer_column_types(
        df: pd.DataFrame, target_column: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """Infer numerical vs categorical columns via dtypes.

        This is a heuristic used for *reporting*. Actual feature typing for
        modeling lives independently in `features.preprocessing`, which can
        override this via explicit lists when needed.
        """
        columns = [c for c in df.columns if c != target_column]
        numerical = df[columns].select_dtypes(include=["number"]).columns.tolist()
        categorical = df[columns].select_dtypes(exclude=["number"]).columns.tolist()
        return numerical, categorical

    def generate_quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """Runs all checks and compiles a DataQualityReport."""
        self.validate_not_empty(df)
        self.validate_target_exists(df)

        errors: List[str] = []
        warnings_found: List[str] = []

        try:
            warnings_found.extend(self.validate_target_values(df))
        except DataValidationError as exc:
            errors.append(str(exc))

        n_rows, n_cols = df.shape
        duplicate_rows = self.check_duplicates(df)
        if duplicate_rows > 0:
            warnings_found.append(f"Found {duplicate_rows} duplicate rows.")

        missing_counts = self.check_missing_values(df)
        missing_pct = {k: (v / n_rows) for k, v in missing_counts.items()}
        high_missing = {k: v for k, v in missing_pct.items() if v > 0.5}
        if high_missing:
            warnings_found.append(
                f"Columns with >50% missing values: {list(high_missing.keys())}"
            )

        numerical_cols, categorical_cols = self.infer_column_types(
            df, self.target_column
        )

        target_distribution = None
        if self.target_column in df.columns:
            target_distribution = (
                df[self.target_column].value_counts(dropna=False).to_dict()
            )

        report = DataQualityReport(
            n_rows=n_rows,
            n_columns=n_cols,
            duplicate_rows=duplicate_rows,
            missing_values=missing_counts,
            missing_percentage=missing_pct,
            numerical_columns=numerical_cols,
            categorical_columns=categorical_cols,
            target_distribution=target_distribution,
            warnings=warnings_found,
            errors=errors,
        )

        if report.has_errors():
            logger.error("Data quality report contains errors: %s", report.errors)
        if report.warnings:
            logger.warning("Data quality warnings: %s", report.warnings)

        return report


def save_quality_report(report: DataQualityReport, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info("Data quality report saved to %s", path)