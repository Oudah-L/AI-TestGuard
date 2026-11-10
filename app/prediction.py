def predict_proba(model, X):
    """Probabilité de fraude (entre 0 et 1) pour chaque transaction."""
    return model.predict_proba(X)[:, 1]


def predict_fraud(model, X, threshold=0.5):
    """Prédit 1 (fraude) si la probabilité dépasse le seuil, sinon 0."""
    return (predict_proba(model, X) >= threshold).astype(int)