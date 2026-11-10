import pandas as pd
from app.preprocessing import split_data, preprocess_data
from app.training import train_model
from app.prediction import predict_proba, predict_fraud
from app.evaluation import evaluate_model

df = pd.read_csv("data/creditcard.csv")
train, val, test = split_data(df)
X_train, y_train = preprocess_data(train)
X_val, y_val = preprocess_data(val)

rows = []
for name in ["logreg", "tree"]:
    model = train_model(X_train, y_train, name)
    proba = predict_proba(model, X_val)
    pred = predict_fraud(model, X_val)
    metrics = evaluate_model(y_val, pred, proba)
    metrics["modele"] = name
    rows.append(metrics)

table = pd.DataFrame(rows).set_index("modele").round(4)
pd.set_option("display.width", 200)
print(table.T)
table.to_csv("reports/comparaison_validation.csv")