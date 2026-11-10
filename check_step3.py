import pandas as pd
from app.preprocessing import split_data, preprocess_data
from app.training import train_model
from app.prediction import predict_proba, predict_fraud

df = pd.read_csv("data/creditcard.csv")
train, val, test = split_data(df)
X_train, y_train = preprocess_data(train)
X_val, y_val = preprocess_data(val)

for name in ["logreg", "tree"]:
    model = train_model(X_train, y_train, name)
    proba = predict_proba(model, X_val)
    pred = predict_fraud(model, X_val)
    print(name, "-> probas entre", proba.min(), "et", proba.max())
    print(name, "-> fraudes détectées sur validation :", int(pred.sum()))