import json
import platform

import numpy as np
import pandas as pd
import sklearn

from app.evaluation import evaluate_model
from app.prediction import predict_fraud, predict_proba
from app.preprocessing import preprocess_data, split_data
from app.training import train_model

SEED = 42
THRESHOLD = 0.99

df = pd.read_csv("data/creditcard.csv")
train, val, test = split_data(df, seed=SEED)
X_train, y_train = preprocess_data(train)
X_val, y_val = preprocess_data(val)

model = train_model(X_train, y_train, "logreg", seed=SEED)
proba = predict_proba(model, X_val)
pred = predict_fraud(model, X_val, threshold=THRESHOLD)
metrics = evaluate_model(y_val, pred, proba)

baseline = {
    "model": "logreg (StandardScaler + LogisticRegression, class_weight=balanced, max_iter=1000)",
    "seed": SEED,
    "threshold": THRESHOLD,
    "split": "60/20/20 stratifié",
    "dataset": {"rows": int(len(df)), "frauds": int(df["Class"].sum())},
    "libraries": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    },
    "validation_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    "tolerances": {"recall_min_drop": 0.03, "precision_min_drop": 0.05,
                   "pr_auc_min_drop": 0.03},
}

with open("reports/baseline.json", "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2, ensure_ascii=False)
print("Baseline enregistrée dans reports/baseline.json")
print(baseline["validation_metrics"])