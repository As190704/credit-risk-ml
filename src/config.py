"""
src/config.py

Centralized configuration for the Explainable Credit Risk Prediction System.

Design principles
------------------
1. No dataset-specific assumptions live here: no feature names, target
   values, or business thresholds are hardcoded. Anything dataset-specific
   (feature lists, target column) is supplied at runtime — either via CLI
   arguments or discovered from the data itself — and only *stored* in
   these dataclasses once known.
2. Every section is a dataclass with __post_init__ validation, so invalid
   configuration fails fast and loudly rather than silently misbehaving.
3. Anything that encodes a business/policy decision (class imbalance
   correction, cost-based thresholds) is opt-in with a safe "do nothing
   special" default.
4. This file has zero dependency on pandas/sklearn/xgboost objects — it is
   pure configuration, safely importable everywhere (including tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Phase 1 — Data loading & validation
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for raw data loading and validation."""

    data_path: str = "data/raw/credit_risk_dataset.csv"

    # Deliberately None: the target column name is dataset-specific and must
    # be supplied explicitly by the caller (CLI arg / notebook cell), never
    # assumed by src/ code.
    target_column: Optional[str] = None

    id_columns: List[str] = field(default_factory=list)
    missing_value_threshold: float = 0.4  # columns above this ratio are flagged
    duplicate_check: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.missing_value_threshold <= 1.0):
            raise ValueError("missing_value_threshold must be within [0, 1].")


# ---------------------------------------------------------------------------
# Phase 2 — Splitting & preprocessing
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Train/test (and optional validation) split configuration."""

    test_size: float = 0.2
    validation_size: float = 0.0  # most model selection uses CVConfig instead
    stratify: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be within (0, 1).")
        if not (0.0 <= self.validation_size < 1.0):
            raise ValueError("validation_size must be within [0, 1).")
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("test_size + validation_size must be < 1.0.")


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline configuration (ColumnTransformer settings)."""

    numerical_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scale_numerical: bool = True
    encode_categorical: str = "onehot"          # "onehot" | "ordinal"
    handle_unknown_categorical: str = "ignore"  # passed to OneHotEncoder

    # Populated at runtime from the discovered/validated schema — never
    # hardcoded with dataset-specific names in src/.
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)

    _VALID_NUMERICAL_STRATEGIES = {"mean", "median", "most_frequent", "constant"}
    _VALID_CATEGORICAL_STRATEGIES = {"most_frequent", "constant"}
    _VALID_ENCODINGS = {"onehot", "ordinal"}

    def __post_init__(self) -> None:
        if self.numerical_impute_strategy not in self._VALID_NUMERICAL_STRATEGIES:
            raise ValueError(
                f"numerical_impute_strategy must be one of "
                f"{self._VALID_NUMERICAL_STRATEGIES}."
            )
        if self.categorical_impute_strategy not in self._VALID_CATEGORICAL_STRATEGIES:
            raise ValueError(
                f"categorical_impute_strategy must be one of "
                f"{self._VALID_CATEGORICAL_STRATEGIES}."
            )
        if self.encode_categorical not in self._VALID_ENCODINGS:
            raise ValueError(f"encode_categorical must be one of {self._VALID_ENCODINGS}.")


# ---------------------------------------------------------------------------
# Reporting (shared across phases)
# ---------------------------------------------------------------------------

@dataclass
class ReportingConfig:
    """Where EDA / evaluation reports and figures are written."""

    reports_dir: str = "reports"
    eda_dir: str = "reports/eda"
    evaluation_dir: str = "reports/evaluation"
    figure_dpi: int = 150
    figure_format: str = "png"


# ---------------------------------------------------------------------------
# Phase 3 — Baseline model training
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model factory configuration for baseline + tuned models."""

    model_type: str = "xgboost"  # "logistic_regression" | "xgboost"
    random_state: int = 42

    # Opt-in only. None = no class-imbalance handling applied. Must never be
    # assumed/auto-enabled by src/ code without explicit caller intent.
    class_imbalance_strategy: Optional[str] = None  # None | "class_weight" | "scale_pos_weight"

    logistic_regression_params: dict = field(default_factory=lambda: {"max_iter": 1000})
    xgboost_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
        }
    )

    _VALID_MODEL_TYPES = {"logistic_regression", "xgboost"}
    _VALID_IMBALANCE_STRATEGIES = {None, "class_weight", "scale_pos_weight"}

    def __post_init__(self) -> None:
        if self.model_type not in self._VALID_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {self._VALID_MODEL_TYPES}.")
        if self.class_imbalance_strategy not in self._VALID_IMBALANCE_STRATEGIES:
            raise ValueError(
                f"class_imbalance_strategy must be one of "
                f"{self._VALID_IMBALANCE_STRATEGIES}."
            )


@dataclass
class CVConfig:
    """Cross-validation configuration used in Phase 3/4."""

    n_splits: int = 5
    shuffle: bool = True
    scoring: str = "roc_auc"
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2.")


# ---------------------------------------------------------------------------
# Phase 4 — Hyperparameter tuning
# ---------------------------------------------------------------------------

@dataclass
class TuningConfig:
    """RandomizedSearchCV configuration for XGBoost tuning."""

    n_iter: int = 50
    cv_splits: int = 5
    scoring: str = "roc_auc"
    n_jobs: int = -1
    random_state: int = 42
    train_val_gap_warning_threshold: float = 0.10  # logged warning, not enforced

    def __post_init__(self) -> None:
        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1.")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be >= 2.")


# ---------------------------------------------------------------------------
# Phase 5 — Threshold optimization
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """Decision threshold selection configuration."""

    strategy: str = "maximize_f1"  # "maximize_f1" | "youden_j" | "cost_based" | "fixed"
    search_start: float = 0.01
    search_stop: float = 0.99
    search_step: float = 0.01

    # cost_based strategy is opt-in and REQUIRES the caller to supply both
    # costs explicitly. No default business cost policy is assumed here.
    cost_fp: Optional[float] = None
    cost_fn: Optional[float] = None

    fixed_threshold: Optional[float] = None  # required only if strategy == "fixed"

    _VALID_STRATEGIES = {"maximize_f1", "youden_j", "cost_based", "fixed"}

    def __post_init__(self) -> None:
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {self._VALID_STRATEGIES}.")
        if not (0.0 <= self.search_start < self.search_stop <= 1.0):
            raise ValueError("search_start/search_stop must satisfy 0 <= start < stop <= 1.")
        if self.strategy == "cost_based" and (self.cost_fp is None or self.cost_fn is None):
            raise ValueError(
                "strategy='cost_based' requires explicit cost_fp and cost_fn; "
                "no default business cost policy is assumed."
            )
        if self.strategy == "fixed" and self.fixed_threshold is None:
            raise ValueError("strategy='fixed' requires fixed_threshold to be set.")
        if self.fixed_threshold is not None and not (0.0 <= self.fixed_threshold <= 1.0):
            raise ValueError("fixed_threshold must be within [0, 1].")


# ---------------------------------------------------------------------------
# Phase 6 — Probability calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Probability calibration configuration."""

    method: str = "sigmoid"     # "sigmoid" (Platt) | "isotonic"
    strategy: str = "prefit"    # "prefit" | "cv"
    calibration_holdout_size: float = 0.25  # fraction of X_train, only used by "prefit"
    cv_splits: int = 5                       # only used by "cv" strategy
    random_state: int = 42

    _VALID_METHODS = {"sigmoid", "isotonic"}
    _VALID_STRATEGIES = {"prefit", "cv"}

    def __post_init__(self) -> None:
        if self.method not in self._VALID_METHODS:
            raise ValueError(f"method must be one of {self._VALID_METHODS}.")
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {self._VALID_STRATEGIES}.")
        if not (0.0 < self.calibration_holdout_size < 1.0):
            raise ValueError("calibration_holdout_size must be within (0, 1).")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be >= 2.")


# ---------------------------------------------------------------------------
# Phase 7 — SHAP explainability
# ---------------------------------------------------------------------------

@dataclass
class ExplainabilityConfig:
    """SHAP global/local explanation configuration."""

    global_sample_size: int = 500     # rows sampled for global SHAP summary
    top_n_local_features: int = 5     # features surfaced per local explanation
    max_display_features: int = 20    # cap on summary/bar plot feature count
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.global_sample_size < 1:
            raise ValueError("global_sample_size must be >= 1.")
        if self.top_n_local_features < 1:
            raise ValueError("top_n_local_features must be >= 1.")
        if self.max_display_features < 1:
            raise ValueError("max_display_features must be >= 1.")


# ---------------------------------------------------------------------------
# Phase 8 — Artifact packaging
# ---------------------------------------------------------------------------

@dataclass
class ArtifactConfig:
    """Paths and naming for versioned artifact packaging."""

    model_name: str = "credit_risk_model"

    models_dir: str = "artifacts/models"
    thresholds_dir: str = "artifacts/thresholds"
    metadata_dir: str = "artifacts/metadata"

    calibration_reports_dir: str = "reports/calibration"
    shap_reports_dir: str = "reports/shap"

    def model_dir(self, version: str) -> str:
        return f"{self.models_dir}/{self.model_name}/{version}"

    def threshold_dir(self, version: str) -> str:
        return f"{self.thresholds_dir}/{version}"

    def metadata_dir_for(self, version: str) -> str:
        return f"{self.metadata_dir}/{version}"


# ---------------------------------------------------------------------------
# Top-level aggregate configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Top-level configuration object composing every phase's config section.

    A single Config instance is threaded through scripts and (in a future
    phase) the API layer, so that every stage of the pipeline reads from one
    consistent, validated source of truth.
    """

    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def validate(self) -> None:
        """
        Cross-section validation that cannot be expressed within a single
        dataclass's __post_init__ (each section already validates itself on
        construction; this checks *interactions* between sections).
        """
        if self.data.target_column is None:
            raise ValueError(
                "Config.data.target_column must be set explicitly before running "
                "any pipeline stage (never inferred/hardcoded)."
            )
        if self.calibration.calibration_holdout_size >= (1.0 - self.split.test_size):
            raise ValueError(
                "calibration_holdout_size, taken out of the training split, must "
                "leave a non-trivial amount of data for model fitting."
            )


def get_default_config() -> Config:
    """Factory returning a fresh default Config (avoids shared mutable state)."""
    return Config()