import json
import os

import pandas as pd
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud, predict_proba
from app.preprocessing import preprocess_data, split_data
from app.training import train_model

BASELINE = "reports/baseline.json"
DATA = "data/creditcard.csv"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(BASELINE) and os.path.exists(DATA)),
    reason="baseline ou données absentes",
)


def test_no_regression_against_baseline():
    with open(BASELINE, encoding="utf-8") as f:
        base = json.load(f)

    df = pd.read_csv(DATA)
    train, val, _ = split_data(df, seed=base["seed"])
    X_train, y_train = preprocess_data(train)
    X_val, y_val = preprocess_data(val)

    model = train_model(X_train, y_train, "logreg", seed=base["seed"])
    proba = predict_proba(model, X_val)
    pred = predict_fraud(model, X_val, threshold=base["threshold"])
    new = evaluate_model(y_val, pred, proba)

    ref = base["validation_metrics"]
    tol = base["tolerances"]
    assert new["recall"] >= ref["recall"] - tol["recall_min_drop"]
    assert new["precision"] >= ref["precision"] - tol["precision_min_drop"]
    assert new["pr_auc"] >= ref["pr_auc"] - tol["pr_auc_min_drop"]