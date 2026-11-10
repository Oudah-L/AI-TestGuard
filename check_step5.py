import pandas as pd
from app.preprocessing import split_data, preprocess_data
from app.training import train_model
from app.prediction import predict_proba, predict_fraud
from app.evaluation import evaluate_model

df = pd.read_csv("data/creditcard.csv")
train, val, test = split_data(df)
X_train, y_train = preprocess_data(train)
X_val, y_val = preprocess_data(val)

model = train_model(X_train, y_train, "logreg")
proba = predict_proba(model, X_val)

rows = []
for t in [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995]:
    pred = predict_fraud(model, X_val, threshold=t)
    m = evaluate_model(y_val, pred, proba)
    rows.append({"seuil": t, "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
                 "precision": round(m["precision"], 3),
                 "recall": round(m["recall"], 3),
                 "f1": round(m["f1"], 3)})

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("reports/seuils_validation.csv", index=False)