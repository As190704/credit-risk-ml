# tests/test_shap.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.config import ModelConfig, PreprocessingConfig
from src.features.preprocessing import FeatureSchema, SchemaValidationError, identify_column_types
from src.models.factory import build_model_pipeline
from src.explainability.shap_explainer import (
    build_feature_name_mapping, build_tree_explainer, transform_for_shap,
)
from src.explainability.global_explanation import generate_global_shap_report
from src.explainability.local_explanation import explain_prediction


def make_data(n=200):
    X, y = make_classification(n_samples=n, n_features=4, n_informative=3, random_state=2)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)])
    rng = np.random.RandomState(2)
    df["employment_type"] = rng.choice(["salaried", "self_employed", "unemployed"], size=n)
    return df, pd.Series(y)


@pytest.fixture
def fitted_context():
    X, y = make_data()
    numerical, categorical = identify_column_types(X)
    pipeline = build_model_pipeline("xgboost", numerical, categorical, PreprocessingConfig(), ModelConfig())
    pipeline.fit(X, y)
    return pipeline, X, y, numerical, categorical


def test_shap_values_match_transformed_feature_count(fitted_context):
    pipeline, X, y, numerical, categorical = fitted_context
    X_transformed, names = transform_for_shap(pipeline, X)
    explainer = build_tree_explainer(pipeline)
    shap_values = explainer(X_transformed)
    assert shap_values.values.shape == (len(X), len(names))


def test_feature_mapping_covers_all_transformed_names(fitted_context):
    pipeline, X, y, numerical, categorical = fitted_context
    _, transformed_names = transform_for_shap(pipeline, X)
    mapping = build_feature_name_mapping(pipeline, numerical, categorical)
    mapped_names = {name for children in mapping.values() for name in children}
    assert mapped_names == set(transformed_names)


def test_global_explanation_runs(fitted_context, tmp_path):
    pipeline, X, y, numerical, categorical = fitted_context
    summary = generate_global_shap_report(pipeline, X, numerical, categorical, tmp_path)
    assert (tmp_path / "shap_summary.png").exists()
    assert (tmp_path / "shap_bar.png").exists()
    assert (tmp_path / "feature_importance.csv").exists()
    assert summary["n_original_features"] == len(numerical) + len(categorical)


def test_local_explanation_structure(fitted_context):
    pipeline, X, y, numerical, categorical = fitted_context
    schema = FeatureSchema(tuple(numerical), tuple(categorical))
    single_row = X.iloc[[0]]

    result = explain_prediction(
        raw_pipeline=pipeline, calibrated_model=pipeline, threshold=0.5,
        input_df=single_row, feature_schema=schema,
        numerical_features=numerical, categorical_features=categorical, top_n=3,
    )
    assert 0.0 <= result["prediction_probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert len(result["top_features"]) <= 3
    for feat in result["top_features"]:
        assert feat["direction"] in ("higher_risk", "lower_risk")


def test_local_explanation_rejects_missing_columns(fitted_context):
    pipeline, X, y, numerical, categorical = fitted_context
    schema = FeatureSchema(tuple(numerical), tuple(categorical))
    bad_row = X.iloc[[0]].drop(columns=[numerical[0]])

    with pytest.raises(SchemaValidationError):
        explain_prediction(
            pipeline, pipeline, 0.5, bad_row, schema, numerical, categorical,
        )


def test_local_explanation_rejects_multi_row_input(fitted_context):
    pipeline, X, y, numerical, categorical = fitted_context
    schema = FeatureSchema(tuple(numerical), tuple(categorical))
    with pytest.raises(ValueError):
        explain_prediction(pipeline, pipeline, 0.5, X.iloc[:2], schema, numerical, categorical)


def test_shap_determinism(fitted_context):
    pipeline, X, y, numerical, categorical = fitted_context
    X_transformed, _ = transform_for_shap(pipeline, X.iloc[:20])
    explainer = build_tree_explainer(pipeline)
    values_1 = explainer(X_transformed).values
    values_2 = explainer(X_transformed).values
    np.testing.assert_array_equal(values_1, values_2)