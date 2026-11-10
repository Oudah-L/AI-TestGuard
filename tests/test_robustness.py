import numpy as np
import pytest

from app.prediction import predict_fraud, predict_proba
from app.preprocessing import preprocess_data


def test_missing_value_is_rejected(small_df):
    bad = small_df.copy()
    bad.loc[0, "V3"] = np.nan
    with pytest.raises(ValueError, match="manquantes"):
        preprocess_data(bad)


def test_infinite_value_is_rejected(small_df):
    bad = small_df.copy()
    bad.loc[0, "V3"] = np.inf
    with pytest.raises(ValueError, match="infinies"):
        preprocess_data(bad)


def test_text_value_is_rejected(small_df):
    bad = small_df.copy()
    bad["V3"] = bad["V3"].astype(str)
    with pytest.raises(ValueError, match="numérique"):
        preprocess_data(bad)


def test_invalid_schema_is_rejected(small_df):
    bad = small_df.drop(columns=["V5"])
    with pytest.raises(ValueError, match="V5"):
        preprocess_data(bad)


def test_extreme_amount_gives_valid_probabilities(model, small_xy):
    X, _ = small_xy
    X_ext = X.copy()
    X_ext["Amount"] = 1e9
    proba = predict_proba(model, X_ext)
    assert np.isfinite(proba).all()
    assert np.all((proba >= 0) & (proba <= 1))


def test_small_noise_changes_probabilities_only_slightly(model, small_xy):
    X, _ = small_xy
    rng = np.random.default_rng(1)
    X_noisy = X.copy()
    # bruit de 1 % de l'écart-type, sur V1 à V28 uniquement
    for c in [f"V{i}" for i in range(1, 29)]:
        X_noisy[c] = X_noisy[c] + rng.normal(0, 0.01 * X[c].std(), len(X))
    diff = np.abs(predict_proba(model, X) - predict_proba(model, X_noisy))
    flips = (predict_fraud(model, X) != predict_fraud(model, X_noisy)).mean()
    print(f"variation moyenne={diff.mean():.4f} max={diff.max():.4f} "
          f"changements de classe={flips:.2%}")
    assert diff.mean() < 0.05
    assert flips < 0.05


def test_distribution_shift_is_measured(model, small_xy):
    X, y = small_xy
    X_shift = X.copy()
    X_shift["V1"] = X_shift["V1"] - 3  # les fraudes se rapprochent des normales
    recall_before = predict_fraud(model, X)[y.to_numpy() == 1].mean()
    recall_after = predict_fraud(model, X_shift)[y.to_numpy() == 1].mean()
    print(f"rappel avant={recall_before:.2f} après décalage={recall_after:.2f}")
    # le décalage doit être détectable : le rappel change
    assert recall_after != recall_before