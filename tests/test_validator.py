import pandas as pd
import pytest

from src.data.validator import DataValidationError, DataValidator


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, None],
            "income": [50000, 60000, 70000, 80000, 90000],
            "employment_type": ["salaried", "self-employed", "salaried", None, "salaried"],
            "target": [0, 1, 0, 1, 0],
        }
    )


def test_missing_target_column_raises():
    df = make_df().drop(columns=["target"])
    validator = DataValidator(target_column="target")
    with pytest.raises(DataValidationError):
        validator.generate_quality_report(df)


def test_empty_dataset_raises():
    validator = DataValidator(target_column="target")
    with pytest.raises(DataValidationError):
        validator.generate_quality_report(pd.DataFrame())


def test_invalid_target_values_are_reported_as_errors():
    df = make_df()
    df.loc[0, "target"] = 99
    validator = DataValidator(target_column="target", valid_target_values=[0, 1])
    report = validator.generate_quality_report(df)
    assert report.has_errors()


def test_missing_value_report():
    df = make_df()
    validator = DataValidator(target_column="target")
    report = validator.generate_quality_report(df)
    assert report.missing_values["age"] == 1
    assert report.missing_values["employment_type"] == 1


def test_duplicate_detection():
    df = pd.concat([make_df(), make_df().iloc[[0]]], ignore_index=True)
    validator = DataValidator(target_column="target")
    report = validator.generate_quality_report(df)
    assert report.duplicate_rows >= 1


def test_column_type_inference():
    numerical, categorical = DataValidator.infer_column_types(make_df(), target_column="target")
    assert "age" in numerical
    assert "income" in numerical
    assert "employment_type" in categorical


def test_binary_target_warning_for_non_binary_target():
    df = make_df()
    df["target"] = [0, 1, 2, 0, 1]
    validator = DataValidator(target_column="target")
    report = validator.generate_quality_report(df)
    assert any("unique values" in w for w in report.warnings)