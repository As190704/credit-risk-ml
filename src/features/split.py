"""
Train/test splitting utilities.

Responsibility: separate features from target and produce reproducible
train/test partitions. No fitting/transformation happens here — that is
handled by `features.preprocessing`, strictly *after* this split, to avoid
leakage.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    excluded_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits a DataFrame into features (X) and target (y).

    `excluded_columns` should come from `DataConfig.excluded_columns` so
    that any column removed from modeling is a documented decision, not a
    silent side effect buried in code.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    excluded_columns = list(excluded_columns or [])
    columns_to_drop = [target_column] + [c for c in excluded_columns if c in df.columns]

    X = df.drop(columns=columns_to_drop)
    y = df[target_column]

    logger.info(
        "Split features/target: X=%s, y=%s, excluded=%s",
        X.shape, y.shape, excluded_columns,
    )
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproducible train/test split.

    Stratification is applied automatically only when the target looks
    categorical (a small number of unique values) and `stratify=True`.
    """
    strat_arg = None
    if stratify:
        n_unique = y.nunique(dropna=True)
        if 1 < n_unique <= 20:
            strat_arg = y
        else:
            logger.warning(
                "Stratification requested but target has %s unique values; "
                "proceeding without stratification.", n_unique,
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat_arg
    )
    logger.info(
        "Train/test split: train=%s, test=%s (test_size=%s, random_state=%s)",
        X_train.shape, X_test.shape, test_size, random_state,
    )
    return X_train, X_test, y_train, y_test