"""
Centralized configuration for the Credit Risk ML pipeline.

Design rationale
-----------------
- Dataclasses give typed, self-documenting configuration objects instead of
  scattering "magic" literals through the codebase.
- Nothing about the *dataset's specific feature names* is hardcoded here.
  Only structural knobs (paths, target column name, split ratios, etc.) are
  configured, since the target dataset has not been finalized yet.
- The target definition is explicit (`target_column`, `valid_target_values`)
  rather than inferred, per the "make target definition explicit" requirement.
- Any column excluded from modeling must be listed in `excluded_columns`
  with a matching entry in `exclusion_reasons` -- this prevents silent,
  undocumented column drops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives and what the target is."""

    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"
    target_column: str = "target"

    # Explicit whitelist of acceptable target values (e.g. [0, 1]). If None,
    # the validator falls back to a generic "binary target" heuristic.
    valid_target_values: Optional[Sequence] = None

    # Columns intentionally excluded from modeling (IDs, post-outcome fields,
    # timestamps, etc). Every exclusion MUST have a documented reason.
    excluded_columns: Tuple[str, ...] = field(default_factory=tuple)
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class PreprocessingConfig:
    numerical_imputation_strategy: str = "median"

    # "constant" preserves a missingness signal as its own category instead
    # of biasing toward the mode -- generally preferable for credit data
    # where "missing" can itself be informative (e.g. no credit history).
    categorical_imputation_strategy: str = "constant"
    categorical_fill_value: str = "missing"

    scale_numerical_features: bool = True  # applied only for linear models
    onehot_handle_unknown: str = "ignore"
    onehot_drop: Optional[str] = None  # e.g. "first" to avoid the dummy trap


@dataclass(frozen=True)
class ReportingConfig:
    eda_output_dir: Path = PROJECT_ROOT / "reports" / "eda"


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def get_default_config() -> Config:
    """Factory instead of a module-level singleton -- keeps the system free
    of hidden global mutable state and makes tests able to build isolated
    configs easily."""
    return Config()