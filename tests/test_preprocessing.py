import numpy as np
import pandas as pd
import pytest

from app.preprocessing import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    preprocess_data,
    split_data,
    validate_data,
)


def make_df(n=10):
    """Fabrique un petit jeu de données valide pour les tests."""
    data = {col: np.linspace(0.0, 1.0, n) for col in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = [0, 1] * (n // 2)
    return pd.DataFrame(data)


def test_valid_data_is_accepted():
    df = make_df()
    assert validate_data(df) is df


def test_empty_data_is_rejected():
    with pytest.raises(ValueError, match="vides"):
        validate_data(make_df().iloc[0:0])


def test_missing_column_is_rejected():
    df = make_df().drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Amount"):
        validate_data(df)


def test_duplicated_column_is_rejected():
    df = make_df()
    df = pd.concat([df, df[["V1"]]], axis=1)
    with pytest.raises(ValueError, match="dupliqu"):
        validate_data(df)


def test_nan_is_rejected():
    df = make_df()
    df.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError, match="manquantes"):
        validate_data(df)


def test_infinite_is_rejected():
    df = make_df()
    df.loc[0, "Amount"] = np.inf
    with pytest.raises(ValueError, match="infinies"):
        validate_data(df)


def test_wrong_type_is_rejected():
    df = make_df()
    df["V1"] = "abc"
    with pytest.raises(ValueError, match="numérique"):
        validate_data(df)


@pytest.mark.parametrize("bad_value", [2, -1, 0.5])
def test_target_other_than_0_1_is_rejected(bad_value):
    df = make_df()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)
    df.loc[0, TARGET_COLUMN] = bad_value
    with pytest.raises(ValueError, match="0 ou 1"):
        validate_data(df)


def test_single_row_is_accepted():
    df = make_df().iloc[[0]]
    X, y = preprocess_data(df)
    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert len(y) == 1


def test_preprocess_dimensions():
    df = make_df(10)
    X, y = preprocess_data(df)
    assert X.shape == (10, len(REQUIRED_COLUMNS) - 1)
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y)


def test_split_keeps_all_rows_without_overlap():
    df = make_df(100)
    train, val, test = split_data(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_split_is_reproducible():
    df = make_df(100)
    a = split_data(df, seed=42)
    b = split_data(df, seed=42)
    assert list(a[0].index) == list(b[0].index)