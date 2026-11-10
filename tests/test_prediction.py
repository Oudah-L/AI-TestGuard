import numpy as np
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud, predict_proba
from app.training import build_model, train_model


def test_probabilities_are_valid(model, small_xy):
    X, _ = small_xy
    proba = predict_proba(model, X)
    assert proba.shape == (len(X),)
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_predictions_are_binary(model, small_xy):
    X, _ = small_xy
    pred = predict_fraud(model, X)
    assert set(np.unique(pred)).issubset({0, 1})
    assert len(pred) == len(X)


def test_threshold_zero_flags_everything_and_one_flags_nothing(model, small_xy):
    X, _ = small_xy
    assert predict_fraud(model, X, threshold=0.0).sum() == len(X)
    assert predict_fraud(model, X, threshold=1.01).sum() == 0


def test_higher_threshold_never_flags_more(model, small_xy):
    X, _ = small_xy
    low = predict_fraud(model, X, threshold=0.3).sum()
    high = predict_fraud(model, X, threshold=0.9).sum()
    assert high <= low


def test_model_detects_obvious_frauds(model, small_xy):
    X, y = small_xy
    pred = predict_fraud(model, X)
    recall = pred[y.to_numpy() == 1].mean()
    assert recall > 0.9


def test_single_row_prediction(model, small_xy):
    X, _ = small_xy
    proba = predict_proba(model, X.iloc[[0]])
    assert proba.shape == (1,)


def test_training_is_reproducible(small_xy):
    X, y = small_xy
    a = predict_proba(train_model(X, y, "tree", seed=42), X)
    b = predict_proba(train_model(X, y, "tree", seed=42), X)
    assert np.allclose(a, b)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        build_model("inexistant")


def test_scaler_is_fitted_on_training_data_only(small_xy):
    X, y = small_xy
    X_train = X.iloc[:200]
    model = train_model(X_train, y.iloc[:200], "logreg")
    scaler = model.named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_train.mean().to_numpy())


def test_evaluate_model_counts_are_consistent():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.4, 0.6, 0.1, 0.2, 0.1])
    m = evaluate_model(y_true, y_pred, y_proba)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 3)
    assert m["TP"] + m["FP"] + m["FN"] + m["TN"] == len(y_true)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(0.25)
    assert m["fnr"] == pytest.approx(0.5)


def test_evaluate_model_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    m = evaluate_model(y, y, y.astype(float))
    assert m["FP"] == 0 and m["FN"] == 0
    assert m["f1"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)