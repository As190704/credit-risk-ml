"""
Data loading utilities.

Responsibility: read raw tabular data from disk into memory with clear,
actionable errors. This module knows nothing about validation rules or
modeling — it only knows how to get data in.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded."""


def load_csv(path: Union[str, Path], **read_csv_kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    path : path to the CSV file.
    read_csv_kwargs : forwarded to pandas.read_csv (e.g. sep, dtype).

    Raises
    ------
    DataLoadError
        If the file does not exist, is a directory, is empty, or fails to parse.
    """
    path = Path(path)

    if not path.exists():
        raise DataLoadError(f"Dataset not found at path: {path}")

    if not path.is_file():
        raise DataLoadError(f"Expected a file but found a directory: {path}")

    try:
        df = pd.read_csv(path, **read_csv_kwargs)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Dataset file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"Failed to parse CSV file {path}: {exc}") from exc

    if df.shape[0] == 0:
        raise DataLoadError(f"Dataset loaded from {path} contains zero rows.")

    logger.info("Loaded dataset from %s with shape %s", path, df.shape)
    return df