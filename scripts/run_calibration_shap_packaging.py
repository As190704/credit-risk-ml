# scripts/run_calibration_shap_packaging.py
"""
Phase 6-8 end-to-end orchestration.

Order of operations (test set is NOT referenced anywhere in this script):
  1. Load + validate data, split (Phase 1-2), reusing X_train/y_train only.
  2. Build XGBoost pipeline (optionally with Phase 4 tuned hyperparameters).
  3. Fit calibrated model (Phase 6) via a leakage-safe prefit holdout split.
  4. Evaluate + plot calibration (Phase 6).
  5. Select threshold on the calibration holdout set (Phase 5, reused).
  6. Generate global SHAP explanation + one example local explanation (Phase 7).
  7. Package everything into a versioned artifact bundle (Phase 8).
  8. Reload the artifact bundle independently and verify it can predict.

The final, one-time test-set evaluation remains the responsibility of
scripts/run_modeling_pipeline.py (Phase 3-5) and is NOT repeated here to
avoid touching the test set a second time.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_default_config
from src.data.loader import load_csv
from src.data.validator import DataValidator
from src.features.split import split_features_target, train_test_split_data
from src.features.preprocessing import FeatureSchema, identify_column_types
from src.models.factory import build_model_pipeline
from src.calibration.calibrator import fit_calibrated_model_with_split
from src.calibration.evaluate import (
    compare_calibration, plot_calibration_curve, plot_probability_distribution, save_calibration_report,
)
from src.optimization.threshold import evaluate_thresholds, select_threshold, save_threshold_artifact, plot_threshold_metrics
from src.explainability.global_explanation import generate_global_shap_report
from src.explainability.local_explanation import explain_prediction
from src.explainability.shap_explainer import build_feature_name_mapping, transform_for_shap
from src.artifacts.save import build_model_metadata, save_model_package
from src.artifacts.load import load_model_artifacts


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("run_calibration_shap_packaging")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--version", type=str, default="v1.0.0")
    args = parser.parse_args()

    config = get_default_config()
    data_path = Path(args.data_path) if args.data_path else config.data.raw_data_path
    target_column = args.target or config.data.target_column

    # --- Phase 1-2 (reused) ---
    df = load_csv(data_path)
    validator = DataValidator(target_column=target_column, valid_target_values=config.data.valid_target_values)
    report = validator.generate_quality_report(df)
    if report.has_errors():
        raise SystemExit(f"Data quality errors: {report.errors}")

    X, y = split_features_target(df, target_column, config.data.excluded_columns)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=config.split.test_size, random_state=config.split.random_state
    )
    numerical_cols, categorical_cols = identify_column_types(X_train)
    logger.info("Test set (%s rows) is NOT used anywhere in this script.", len(X_test))

    # --- Base pipeline (ideally hyperparameters from Phase 4's saved best_params) ---
    base_pipeline = build_model_pipeline(
        "xgboost", numerical_cols, categorical_cols, config.preprocessing, config.model, y_train=y_train
    )

    # --- Phase 6: calibration ---
    fit_result, X_calib, y_calib = fit_calibrated_model_with_split(
        base_pipeline, X_train, y_train, config.calibration
    )
    proba_before = fit_result.raw_pipeline.predict_proba(X_calib)[:, 1]
    proba_after = fit_result.calibrated_model.predict_proba(X_calib)[:, 1]

    calibration_comparison = compare_calibration(y_calib.to_numpy(), proba_before, proba_after)
    calib_dir = config.artifacts.calibration_reports_dir
    plot_calibration_curve(y_calib, proba_before, proba_after, calib_dir / "calibration_curve.png")
    plot_probability_distribution(proba_before, proba_after, calib_dir / "probability_distribution.png")
    save_calibration_report(calibration_comparison, calib_dir / "calibration_metrics.json")
    logger.info("Calibration comparison: %s", calibration_comparison)

    # --- Phase 5 (reused): threshold selection on the calibration holdout set ---
    threshold_table = evaluate_thresholds(y_calib.to_numpy(), proba_after)
    threshold_result = select_threshold(threshold_table, config.threshold.strategy)
    plot_threshold_metrics(
        threshold_table, config.artifacts.threshold_reports_dir / "threshold_metrics.png", threshold_result.threshold
    )
    save_threshold_artifact(threshold_result, threshold_table, config.artifacts.threshold_reports_dir)
    logger.info("Selected threshold: %.2f", threshold_result.threshold)

    # --- Phase 7: SHAP ---
    shap_summary = generate_global_shap_report(
        fit_result.raw_pipeline, X_calib, numerical_cols, categorical_cols,
        config.artifacts.shap_reports_dir / "global",
        max_display=config.explainability.max_display_features,
        sample_size=config.explainability.global_sample_size,
        random_state=config.explainability.random_state,
    )
    logger.info("Global SHAP summary: %s original features explained.", shap_summary["n_original_features"])

    feature_schema = FeatureSchema(tuple(numerical_cols), tuple(categorical_cols))
    example_row = X_calib.iloc[[0]]
    local_explanation = explain_prediction(
        fit_result.raw_pipeline, fit_result.calibrated_model, threshold_result.threshold,
        example_row, feature_schema, numerical_cols, categorical_cols,
        top_n=config.explainability.top_n_local_features,
    )
    logger.info("Example local explanation: %s", local_explanation)

    # --- Phase 8: packaging ---
    _, transformed_names = transform_for_shap(fit_result.raw_pipeline, X_calib.iloc[:1])
    feature_mapping = build_feature_name_mapping(fit_result.raw_pipeline, numerical_cols, categorical_cols)
    shap_metadata = {
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "transformed_feature_names": transformed_names,
        "feature_mapping": feature_mapping,
    }

    metadata = build_model_metadata(
        model_name=config.artifacts.model_name,
        model_version=args.version,
        model_type=type(fit_result.raw_pipeline.named_steps["classifier"]).__name__,
        calibration_method=config.calibration.method,
        calibration_strategy=config.calibration.strategy,
        threshold=threshold_result.threshold,
        threshold_strategy=config.threshold.strategy,
        training_random_state=config.model.random_state,
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
        target_column=target_column,
        best_hyperparameters=config.model.xgboost_params,
        evaluation_metrics=calibration_comparison["after_calibration"],
        calibration_metrics={
            "brier_score_delta": calibration_comparison["brier_score_delta"],
            "log_loss_delta": calibration_comparison["log_loss_delta"],
        },
    )

    threshold_payload = threshold_result.to_dict()
    save_model_package(
        args.version, fit_result.calibrated_model, fit_result.raw_pipeline,
        shap_metadata, threshold_payload, metadata, config.artifacts,
    )

    # --- Verify independent reload (no training code required) ---
    reloaded = load_model_artifacts(args.version, config.artifacts)
    sample_pred = reloaded.predict(X_calib.iloc[:5])
    logger.info("Reloaded artifact sample predictions: %s", sample_pred)
    logger.info("Phase 6-8 pipeline complete. Version=%s", args.version)


if __name__ == "__main__":
    main()