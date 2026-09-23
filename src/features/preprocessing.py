"""
Preprocessing pipeline construction.

Responsibility: turn raw feature columns into model-ready numeric arrays
using scikit-learn Pipeline/ColumnTransformer, fitted ONLY on training data.
Supports Logistic Regression (linear) and XGBoost (tree) requirements via a
single configurable factory, avoiding duplicated pipeline code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PreprocessingConfig

logger = logging.getLogger(__name__)

ModelFamily = Literal["linear", "tree"]


class SchemaValidationError(Exception):
    """Raised when a new dataset's schema is incompatible with training schema."""


@dataclass(frozen=True)
class FeatureSchema:
    """Captures the column layout used to fit a preprocessor, so future
    inference-time data can be validated for consistency before transform."""

    numerical_features: Tuple[str, ...]
    categorical_features: Tuple[str, ...]

    @property
    def all_features(self) -> Tuple[str, ...]:
        return tuple(self.numerical_features) + tuple(self.categorical_features)

    def validate(self, df: pd.DataFrame) -> None:
        missing = set(self.all_features) - set(df.columns)
        if missing:
            raise SchemaValidationError(
                f"Input data is missing expected columns: {sorted(missing)}"
            )
        extra = set(df.columns) - set(self.all_features)
        if extra:
            logger.warning(
                "Input data has extra columns not seen during training "
                "(they will be ignored): %s", sorted(extra),
            )


def identify_column_types(
    X: pd.DataFrame,
    numeric_override: Optional[Sequence[str]] = None,
    categorical_override: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Identify numerical vs categorical columns from dtypes.

    Overrides let callers correct heuristics (e.g. a numeric-coded
    categorical ID column) without hardcoding dataset-specific names inside
    this module.
    """
    if numeric_override is not None or categorical_override is not None:
        numerical = list(numeric_override or [])
        categorical = list(categorical_override or [])
        remaining = [c for c in X.columns if c not in numerical and c not in categorical]
        if remaining:
            raise ValueError(
                f"Columns not classified by overrides: {remaining}. Provide "
                "complete numeric_override/categorical_override lists."
            )
        return numerical, categorical

    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical, categorical


def build_numerical_pipeline(config: PreprocessingConfig, model_family: ModelFamily) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy=config.numerical_imputation_strategy))]
    # Scaling matters for gradient-based linear models (Logistic Regression)
    # but is unnecessary for tree-based models (XGBoost), which are
    # invariant to monotonic feature scaling.
    if model_family == "linear" and config.scale_numerical_features:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_categorical_pipeline(config: PreprocessingConfig) -> Pipeline:
    imputer_kwargs = {"strategy": config.categorical_imputation_strategy}
    if config.categorical_imputation_strategy == "constant":
        imputer_kwargs["fill_value"] = config.categorical_fill_value

    steps = [
        ("imputer", SimpleImputer(**imputer_kwargs)),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown=config.onehot_handle_unknown,
                drop=config.onehot_drop,
                sparse_output=False,
            ),
        ),
    ]
    return Pipeline(steps)


def build_preprocessor(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    config: PreprocessingConfig,
    model_family: ModelFamily = "linear",
) -> ColumnTransformer:
    """Builds a ColumnTransformer suited to the given model family.

    IMPORTANT: the returned transformer is UNFITTED. Callers must fit on
    the *training* split only, then `.transform` on validation/test/
    inference data, to avoid data leakage.
    """
    transformers = []
    if numerical_features:
        transformers.append(
            ("numerical", build_numerical_pipeline(config, model_family), list(numerical_features))
        )
    if categorical_features:
        transformers.append(
            ("categorical", build_categorical_pipeline(config), list(categorical_features))
        )

    if not transformers:
        raise ValueError("No numerical or categorical features provided to build_preprocessor.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fits the preprocessor on TRAINING data only.

    Never call this with combined train+test data — doing so leaks test-set
    statistics (e.g. imputation medians, scaler mean/std, seen categories)
    into training.
    """
    preprocessor.fit(X_train)
    logger.info(
        "Preprocessor fitted on training data: %s rows, %s output features",
        X_train.shape[0], len(get_feature_names_out(preprocessor)),
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """Convenience wrapper, valid only after the preprocessor is fitted."""
    return list(preprocessor.get_feature_names_out())


def save_preprocessor(preprocessor: ColumnTransformer, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info("Preprocessor saved to %s", path)


def load_preprocessor(path: Union[str, Path]) -> ColumnTransformer:
    return joblib.load(Path(path))