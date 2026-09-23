# scripts/run_modeling_pipeline.py
"""
Phase 3-5 end-to-end orchestration.

Order of operations (test set is touched exactly once, at the very end):
  1. Load + validate data (Phase 1).
  2. Split features/target, train/test split (Phase 2).
  3. Baseline CV for Logistic Regression and XGBoost (Phase 3).
  4. Build model comparison report (Phase 3).
  5. RandomizedSearchCV tuning of XGBoost (Phase 4).
  6. Out-of-fold validation probabilities from the TUNED pipeline (Phase 5).
  7. Threshold selection (Phase 5).
  8. ONE-TIME final evaluation on the held-out test set.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_default_config
from src.data.loader import load_csv
from src.data.validator import DataValidator
from src.features.split import split_features_target, train_test_split_data
from src.features.preprocessing import identify_column_types
from src.models.factory import build_model_pipeline
from src.models.train import cross_validate_model, fit_final_pipeline, get_validation_probabilities
from src.models.evaluate import (
    ModelComparisonEntry, build_comparison_report, compute_classification_metrics,
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve, compute_confusion_matrix,
    save_evaluation_report,
)
from src.models.tune import run_randomized_search, save_tuning_artifacts
from src.optimization.threshold import (
    evaluate_thresholds, select_threshold, plot_threshold_metrics,
    plot_precision_recall_tradeoff, save_threshold_artifact, apply_threshold,
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("run_modeling_pipeline")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--tuning-n-iter", type=int, default=None)
    args = parser.parse_args()

    config = get_default_config()
    data_path = Path(args.data_path) if args.data_path else config.data.raw_data_path
    target_column = args.target or config.data.target_column
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # --- Phase 1: load + validate ---
    df = load_csv(data_path)
    validator = DataValidator(target_column=target_column, valid_target_values=config.data.valid_target_values)
    report = validator.generate_quality_report(df)
    if report.has_errors():
        raise SystemExit(f"Data quality errors: {report.errors}")

    # --- Phase 2: split ---
    X, y = split_features_target(df, target_column, config.data.excluded_columns)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=config.split.test_size, random_state=config.split.random_state
    )
    numerical_cols, categorical_cols = identify_column_types(X_train)
    logger.info("Test set reserved: %s rows (touched only at the final evaluation step).", len(X_test))

    # --- Phase 3: baseline CV for both models ---
    comparison_entries = []
    for model_name in ["logistic_regression", "xgboost"]:
        pipeline = build_model_pipeline(
            model_name, numerical_cols, categorical_cols, config.preprocessing, config.model, y_train=y_train
        )
        cv_result = cross_validate_model(model_name, pipeline, X_train, y_train, config.cv)
        comparison_entries.append(
            ModelComparisonEntry(
                model_name=model_name,
                cv_mean_metrics=cv_result.mean_metrics,
                cv_std_metrics=cv_result.std_metrics,
                validation_metrics=None,
                training_time_seconds=cv_result.training_time_seconds,
                model_config=config.model.__dict__,
                limitations=["Baseline hyperparameters, not yet tuned."],
            )
        )

    comparison_df = build_comparison_report(comparison_entries)
    config.artifacts.evaluation_reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(config.artifacts.evaluation_reports_dir / f"model_comparison_{version}.csv", index=False)
    logger.info("Baseline model comparison:\n%s", comparison_df.to_string())

    # --- Phase 4: tune XGBoost ---
    tuning_config = config.tuning
    if args.tuning_n_iter:
        tuning_config = config.tuning.__class__(**{**config.tuning.__dict__, "n_iter": args.tuning_n_iter})

    xgb_pipeline = build_model_pipeline(
        "xgboost", numerical_cols, categorical_cols, config.preprocessing, config.model, y_train=y_train
    )
    tuning_result = run_randomized_search(xgb_pipeline, X_train, y_train, tuning_config)
    save_tuning_artifacts(tuning_result, config.artifacts.models_dir, config.artifacts.metadata_dir, version)

    # --- Phase 5: threshold optimization (training data only) ---
    y_val_true, y_val_proba = get_validation_probabilities(
        tuning_result.best_estimator, X_train, y_train, config.cv,
        strategy=config.threshold.validation_strategy, holdout_size=config.threshold.holdout_size,
    )
    threshold_table = evaluate_thresholds(y_val_true, y_val_proba)

    strategy_kwargs = {}
    if config.threshold.strategy == "min_recall":
        strategy_kwargs = {"min_recall": config.threshold.min_recall}
    elif config.threshold.strategy == "min_precision":
        strategy_kwargs = {"min_precision": config.threshold.min_precision}
    elif config.threshold.strategy == "cost_based":
        strategy_kwargs = {"cost_fp": config.threshold.cost_fp, "cost_fn": config.threshold.cost_fn}

    threshold_result = select_threshold(threshold_table, config.threshold.strategy, **strategy_kwargs)
    logger.info("Selected threshold: %.2f (%s)", threshold_result.threshold, threshold_result.rationale)

    threshold_dir = config.artifacts.thresholds_dir
    save_threshold_artifact(
        threshold_result, threshold_table, threshold_dir,
        metadata={"version": version, "validation_strategy": config.threshold.validation_strategy},
    )
    plots_dir = config.artifacts.threshold_reports_dir / version
    plot_threshold_metrics(threshold_table, plots_dir / "threshold_metrics.png", threshold_result.threshold)
    plot_precision_recall_tradeoff(threshold_table, plots_dir / "pr_tradeoff.png", threshold_result.threshold)

    # --- FINAL: fit on full training set, evaluate on test set EXACTLY ONCE ---
    final_pipeline = fit_final_pipeline(tuning_result.best_estimator, X_train, y_train)
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = apply_threshold(y_test_proba, threshold_result.threshold)

    final_metrics = compute_classification_metrics(y_test.to_numpy(), y_test_pred, y_test_proba)
    logger.info("FINAL TEST SET METRICS (touched once): %s", final_metrics)

    eval_dir = config.artifacts.evaluation_reports_dir / version
    plot_confusion_matrix(compute_confusion_matrix(y_test, y_test_pred), eval_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_test_proba, eval_dir / "roc_curve.png")
    plot_pr_curve(y_test, y_test_proba, eval_dir / "pr_curve.png")
    save_evaluation_report(
        {"final_test_metrics": final_metrics, "selected_threshold": threshold_result.threshold,
         "best_params": tuning_result.best_params, "version": version},
        eval_dir / "final_test_evaluation.json",
    )


if __name__ == "__main__":
    main()