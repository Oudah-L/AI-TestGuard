import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_data(df):
    """Vérifie que les données respectent le contrat. Lève ValueError sinon."""
    if df.empty:
        raise ValueError("Les données sont vides.")

    if df.columns.duplicated().any():
        raise ValueError("Des colonnes sont dupliquées.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"La colonne {col} doit être numérique.")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Valeurs manquantes détectées.")

    if np.isinf(df[REQUIRED_COLUMNS]).any().any():
        raise ValueError("Valeurs infinies détectées.")

    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("La colonne Class doit contenir uniquement 0 ou 1.")

    return df


def preprocess_data(df):
    """Sépare les variables (X) de la cible (y)."""
    validate_data(df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def split_data(df, seed=42):
    """Coupe en train 60 % / validation 20 % / test 20 %, avec stratification."""
    train, temp = train_test_split(
        df, test_size=0.4, stratify=df[TARGET_COLUMN], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp[TARGET_COLUMN], random_state=seed
    )
    return train, val, test