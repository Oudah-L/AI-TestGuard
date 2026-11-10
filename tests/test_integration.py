import joblib
import numpy as np
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud, predict_proba
from app.preprocessing import preprocess_data, split_data
from app.training import train_model


def test_pipeline_end_to_end(small_df, tmp_path):
    # 1. séparation avant toute transformation apprise
    train, _val, test = split_data(small_df)
    X_train, y_train = preprocess_data(train)
    X_test, y_test = preprocess_data(test)

    # 2. entraînement sur le train uniquement
    model = train_model(X_train, y_train, "logreg")

    # 3. sauvegarde puis rechargement
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)

    # 4. prédictions identiques avant / après rechargement
    before = predict_proba(model, X_test)
    after = predict_proba(reloaded, X_test)
    assert np.allclose(before, after)
    assert np.array_equal(
        predict_fraud(model, X_test), predict_fraud(reloaded, X_test)
    )

    # 5. dimensions cohérentes
    assert len(after) == len(X_test)

    # 6. rapport de métriques produit
    metrics = evaluate_model(y_test, predict_fraud(reloaded, X_test), after)
    assert 0.0 <= metrics["recall"] <= 1.0


def test_no_non_finite_values_after_scaling(small_df):
    train, _, _ = split_data(small_df)
    X_train, y_train = preprocess_data(train)
    model = train_model(X_train, y_train, "logreg")
    scaled = model.named_steps["scaler"].transform(X_train)
    assert np.isfinite(scaled).all()
    assert scaled.shape == X_train.shape


def test_scaler_does_not_learn_from_test_set(small_df):
    train, _, test = split_data(small_df)
    X_train, y_train = preprocess_data(train)
    X_test, _ = preprocess_data(test)
    model = train_model(X_train, y_train, "logreg")
    scaler = model.named_steps["scaler"]
    # le scaler correspond au train, pas au test
    assert np.allclose(scaler.mean_, X_train.mean().to_numpy())
    assert not np.allclose(scaler.mean_, X_test.mean().to_numpy())


def test_loading_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        joblib.load(tmp_path / "inexistant.joblib")


def test_invalid_input_is_rejected_before_prediction(small_df):
    bad = small_df.drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Amount"):
        preprocess_data(bad)