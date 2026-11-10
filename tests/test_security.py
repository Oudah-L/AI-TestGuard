import re
from pathlib import Path

import pytest

from app.preprocessing import preprocess_data


def test_extra_column_is_ignored_not_used(small_df):
    df = small_df.copy()
    df["colonne_inattendue"] = 1.0
    X, _ = preprocess_data(df)
    assert "colonne_inattendue" not in X.columns


def test_missing_column_is_rejected(small_df):
    with pytest.raises(ValueError, match="V7"):
        preprocess_data(small_df.drop(columns=["V7"]))


def test_wrong_type_is_rejected(small_df):
    df = small_df.copy()
    df["Amount"] = "beaucoup"
    with pytest.raises(ValueError, match="numérique"):
        preprocess_data(df)


def test_no_secrets_in_python_files():
    pattern = re.compile(
        r"(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]+['\"]",
        re.IGNORECASE,
    )
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for folder in ["app", "tests"]:
        for path in (root / folder).rglob("*.py"):
            if path.name == "test_security.py":
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path))
    assert offenders == []