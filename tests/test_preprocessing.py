import pandas as pd
import pytest

from src.config import PreprocessingConfig
from src.features.preprocessing import (
    FeatureSchema,
    SchemaValidationError,
    build_preprocessor,
    fit_preprocessor,
    identify_column_types,
)
from src.features.split import split_features_target, train_test_split_data


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "income": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
            "employment_type": [
                "salaried", "self-employed", "salaried", "unemployed",
                "salaried", "self-employed", "salaried", "unemployed",
            ],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_identify_column_types():
    X, _ = split_features_target(make_data(), target_column="target")
    numerical, categorical = identify_column_types(X)
    assert set(numerical) == {"age", "income"}
    assert set(categorical) == {"employment_type"}


def test_train_test_split_reproducibility():
    X, y = split_features_target(make_data(), target_column="target")
    split1 = train_test_split_data(X, y, test_size=0.25, random_state=42)
    split2 = train_test_split_data(X, y, test_size=0.25, random_state=42)
    pd.testing.assert_frame_equal(split1[0], split2[0])
    pd.testing.assert_series_equal(split1[2], split2[2])


def test_preprocessor_fit_transform_linear():
    X, y = split_features_target(make_data(), target_column="target")
    X_train, X_test, _, _ = train_test_split_data(X, y, test_size=0.25, random_state=42)
    numerical, categorical = identify_column_types(X_train)
    config = PreprocessingConfig()

    preprocessor = build_preprocessor(numerical, categorical, config, model_family="linear")
    fit_preprocessor(preprocessor, X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    assert X_train_t.shape[0] == X_train.shape[0]
    assert X_test_t.shape[0] == X_test.shape[0]
    # Scaled numerical column should have ~zero mean on the data it was fit on
    assert abs(X_train_t[:, 0].mean()) < 1e-6


def test_preprocessor_no_scaling_for_tree_model():
    X, _ = split_features_target(make_data(), target_column="target")
    numerical, categorical = identify_column_types(X)
    config = PreprocessingConfig()

    preprocessor = build_preprocessor(numerical, categorical, config, model_family="tree")
    fit_preprocessor(preprocessor, X)
    transformed = preprocessor.transform(X)

    # Without scaling, raw "age" magnitude should be preserved
    assert transformed[:, 0].max() > 1


def test_unknown_category_handled_gracefully():
    X, _ = split_features_target(make_data(), target_column="target")
    numerical, categorical = identify_column_types(X)
    config = PreprocessingConfig()

    preprocessor = build_preprocessor(numerical, categorical, config, model_family="linear")
    fit_preprocessor(preprocessor, X)

    X_new = X.iloc[:2].copy()
    X_new["employment_type"] = "freelancer"  # unseen category

    transformed = preprocessor.transform(X_new)  # must not raise
    assert transformed.shape[0] == 2


def test_schema_validation_detects_missing_columns():
    X, _ = split_features_target(make_data(), target_column="target")
    numerical, categorical = identify_column_types(X)
    schema = FeatureSchema(tuple(numerical), tuple(categorical))

    bad_df = X.drop(columns=["income"])
    with pytest.raises(SchemaValidationError):
        schema.validate(bad_df)


def test_no_leakage_preprocessor_uses_train_statistics_only():
    X, y = split_features_target(make_data(), target_column="target")
    X_train, X_test, _, _ = train_test_split_data(X, y, test_size=0.5, random_state=1)
    numerical, categorical = identify_column_types(X_train)
    config = PreprocessingConfig()

    preprocessor = build_preprocessor(numerical, categorical, config, model_family="linear")
    fit_preprocessor(preprocessor, X_train)

    scaler = preprocessor.named_transformers_["numerical"].named_steps["scaler"]
    expected_mean = X_train["age"].mean()
    assert abs(scaler.mean_[0] - expected_mean) < 1e-6