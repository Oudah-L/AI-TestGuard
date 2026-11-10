from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_model(name="logreg", seed=42):
    """Construit un pipeline (scaler + modèle) non entraîné."""
    if name == "logreg":
        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed
        )
    elif name == "tree":
        clf = DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, random_state=seed
        )
    else:
        raise ValueError(f"Modèle inconnu : {name}")

    return Pipeline([("scaler", StandardScaler()), ("model", clf)])


def train_model(X, y, name="logreg", seed=42):
    """Entraîne le pipeline sur X, y (données d'entraînement uniquement)."""
    model = build_model(name, seed)
    model.fit(X, y)
    return model