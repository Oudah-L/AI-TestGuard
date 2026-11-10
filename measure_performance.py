import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import psutil

from app.prediction import predict_proba
from app.preprocessing import preprocess_data, split_data
from app.training import train_model

df = pd.read_csv("data/creditcard.csv")
train, val, _ = split_data(df)
X_train, y_train = preprocess_data(train)
X_val, _ = preprocess_data(val)

model = train_model(X_train, y_train, "logreg")
joblib.dump(model, "reports/model_logreg.joblib")
size_kb = os.path.getsize("reports/model_logreg.joblib") / 1024

# Latence d'une prédiction isolée : 1000 répétitions
times = []
for i in range(1000):
    row = X_val.iloc[[i]]
    t0 = time.perf_counter()
    predict_proba(model, row)
    times.append((time.perf_counter() - t0) * 1000)
times = np.array(times)

# Débit en lot : toute la validation
t0 = time.perf_counter()
predict_proba(model, X_val)
batch_s = time.perf_counter() - t0

mem_mb = psutil.Process().memory_info().rss / 1024 / 1024

result = {
    "materiel": f"{psutil.cpu_count(logical=True)} coeurs logiques, "
                f"{psutil.virtual_memory().total / 1024**3:.1f} Go RAM",
    "repetitions_prediction_isolee": 1000,
    "latence_moyenne_ms": round(float(times.mean()), 3),
    "latence_p95_ms": round(float(np.percentile(times, 95)), 3),
    "lot_taille": len(X_val),
    "lot_duree_s": round(batch_s, 4),
    "debit_predictions_par_s": int(len(X_val) / batch_s),
    "taille_modele_ko": round(size_kb, 1),
    "memoire_processus_mo": round(mem_mb, 1),
}
with open("reports/performance.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
for k, v in result.items():
    print(k, ":", v)