import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay)
from app.preprocessing import split_data, preprocess_data
from app.training import train_model
from app.prediction import predict_proba, predict_fraud

THRESHOLD = 0.99

df = pd.read_csv("data/creditcard.csv")
train, val, test = split_data(df)
X_train, y_train = preprocess_data(train)
X_val, y_val = preprocess_data(val)

fig_roc, ax_roc = plt.subplots()
fig_pr, ax_pr = plt.subplots()

for name, label in [("logreg", "Régression logistique"), ("tree", "Arbre")]:
    model = train_model(X_train, y_train, name)
    proba = predict_proba(model, X_val)
    RocCurveDisplay.from_predictions(y_val, proba, name=label, ax=ax_roc)
    PrecisionRecallDisplay.from_predictions(y_val, proba, name=label, ax=ax_pr)

    if name == "logreg":
        pred = predict_fraud(model, X_val, threshold=THRESHOLD)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_val, pred, display_labels=["Normale", "Fraude"], ax=ax_cm)
        ax_cm.set_title(f"Matrice de confusion (seuil {THRESHOLD})")
        fig_cm.savefig("reports/matrice_confusion.png", dpi=120)

ax_roc.set_title("Courbes ROC (validation)")
ax_pr.set_title("Courbes précision-rappel (validation)")
fig_roc.savefig("reports/courbe_roc.png", dpi=120)
fig_pr.savefig("reports/courbe_precision_rappel.png", dpi=120)
print("Graphiques enregistrés dans reports/")