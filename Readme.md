# Credit Risk ML — Phase 1 & 2 Foundation

Explainable Credit Risk Prediction System — data ingestion, validation, EDA,
splitting, and preprocessing foundation. Model training, calibration,
SHAP, API, and deployment are intentionally **out of scope** for this phase.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Phase 3-5: Modeling, Tuning, Threshold Optimization

Run the full pipeline (baseline CV -> XGBoost tuning -> threshold selection
-> one-time test evaluation):

    python scripts/run_modeling_pipeline.py --data-path data/raw/your_dataset.csv --target target

Outputs:
- `reports/evaluation/model_comparison_<version>.csv` - LR vs XGBoost baseline CV comparison
- `artifacts/models/xgboost_tuned_<version>.joblib` - tuned XGBoost pipeline
- `artifacts/metadata/xgboost_tuning_*` - best params, cv_results, gap diagnostics
- `artifacts/thresholds/threshold.json` - selected threshold + rationale (loadable by future inference service)
- `reports/threshold_analysis/<version>/*.png` - threshold/precision-recall plots
- `reports/evaluation/<version>/final_test_evaluation.json` - ONE-TIME held-out test evaluation

### Key decisions
- Every model is a single `Pipeline(preprocessor + classifier)`; CV/tuning
  clone-and-refit this pipeline per fold, so preprocessing statistics never
  leak across folds or into the test set.
- ROC AUC is the primary tuning metric (threshold-independent, ranking-focused);
  PR AUC/precision/recall/F1 are reported as complementary diagnostics.
- Class imbalance handling (`class_weight="balanced"` / `scale_pos_weight`)
  is opt-in via `ModelConfig.class_imbalance_strategy`, never automatic.
- Threshold selection strategy is configurable (`maximize_f1`, `min_recall`,
  `min_precision`, `cost_based`) and evaluated only on training/OOF/holdout
  data -- the test set is used exactly once, at the very end, for reporting only.
- XGBoost native early stopping was deliberately NOT integrated into the
  RandomizedSearchCV loop (see `src/models/tune.py` docstring) due to
  preprocessing-refit complications; regularization + train/val gap
  monitoring are used instead.

### Assumptions & Limitations (Phase 3-5)
- Model outputs (probabilities and thresholded labels) are decision-support
  signals, not automated final lending decisions.
- Default hyperparameter search space in `TuningConfig` is a general-purpose
  starting point; should be revisited once the dataset size/shape is finalized.
- `cost_based` threshold strategy requires real cost inputs from business
  stakeholders; no default costs are assumed.
- Probability calibration (Phase 6) and SHAP explainability (Phase 7) are
  intentionally out of scope here — `Pipeline`/`XGBClassifier` outputs from
  this phase are the direct inputs to those future phases.



## Phase 6-8: Calibration, Explainability, Packaging

Run:
    python scripts/run_calibration_shap_packaging.py --data-path data/raw/your_dataset.csv --target target --version v1.0.0

### Key decisions
- Calibration uses a leakage-safe "prefit" holdout split by default:
  X_train -> X_fit (base model) / X_calib (sigmoid calibration + threshold
  selection). A "cv" strategy (CalibratedClassifierCV internal k-fold) is
  also implemented and documented as a more data-efficient alternative.
- Discrimination (ROC AUC/PR AUC), calibration (Brier score/log loss), and
  thresholding are treated as three distinct concepts throughout -- never
  conflated in code, metrics, or reporting.
- SHAP's TreeExplainer is built ONLY from the uncalibrated raw XGBoost
  pipeline, never from the CalibratedClassifierCV wrapper. Every
  explanation output documents this distinction explicitly.
- Feature-name mapping for one-hot encoded features is derived from the
  fitted preprocessor's `get_feature_names_out()` plus the known
  numerical/categorical column lists -- never dataset-specific hardcoding.
- Packaging bundles exactly two model artifacts (`model.joblib`,
  `raw_pipeline.joblib`) rather than three+, since both already fully
  encapsulate preprocessing; this avoids redundant, driftable copies.
- Versioning: semantic (`vMAJOR.MINOR.PATCH`). Bump MAJOR for feature
  schema/input-contract changes, MINOR for model/preprocessing/calibration
  changes, PATCH for threshold-only or metadata updates.

### Security
- joblib artifacts are only ever loaded from this repository's own
  `artifacts/` tree. No URL-based or remote artifact loading is
  implemented. Model metadata never stores applicant-level data, only
  column names, aggregate metrics, and environment/versioning info.

### Assumptions & Limitations (Phase 6-8)
- "prefit" calibration spends ~25% of training data on calibration/
  threshold selection rather than model fitting -- configurable via
  `CalibrationConfig.calibration_holdout_size`.
- SHAP values are computed in the raw XGBoost model's log-odds space, not
  probability space, and are explicitly documented as not equal to a
  decomposition of the final calibrated probability.
- SHAP one-hot aggregation uses summation across a categorical feature's
  encoded columns -- a standard, additive-property-preserving choice.
- No artifact registry/index (e.g. "latest" pointer) yet; version strings
  must be known by the caller. Reserved for Phase 9.