import numpy as np
import pandas as pd
import pytest

from app.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, preprocess_data
from app.training import train_model


@pytest.fixture(scope="session")
def small_df():
    """300 transactions artificielles, dont 30 fraudes décalées sur V1 et V2."""
    rng = np.random.default_rng(0)
    n = 300
    data = {col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS}
    y = np.array([1] * 30 + [0] * 270)
    data["V1"] = data["V1"] + 4 * y
    data["V2"] = data["V2"] - 4 * y
    data["Amount"] = np.abs(data["Amount"]) * 50
    data[TARGET_COLUMN] = y
    return pd.DataFrame(data).sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="session")
def small_xy(small_df):
    return preprocess_data(small_df)


@pytest.fixture(scope="session")
def model(small_xy):
    X, y = small_xy
    return train_model(X, y, "logreg")