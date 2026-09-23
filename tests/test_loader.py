import pandas as pd
import pytest

from src.data.loader import DataLoadError, load_csv


def test_load_valid_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)

    df = load_csv(csv_path)

    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_load_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(DataLoadError):
        load_csv(missing_path)


def test_load_empty_file(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")
    with pytest.raises(DataLoadError):
        load_csv(empty_path)


def test_load_directory_instead_of_file(tmp_path):
    with pytest.raises(DataLoadError):
        load_csv(tmp_path)