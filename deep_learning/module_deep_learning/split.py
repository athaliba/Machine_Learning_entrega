"""Separação 60% treino, 20% validação e 20% teste, como no notebook de referência."""

from sklearn.model_selection import train_test_split

from module_deep_learning.config import SEED
from module_deep_learning.features import FEATURES, TARGET


def split_data(data):
    X, y = data[FEATURES].copy(), data[TARGET].copy()
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEED,
    )
    for label, target in [("treino", y_train), ("validação", y_val), ("teste", y_test)]:
        if target.nunique() != 2:
            raise ValueError(f"O conjunto de {label} precisa conter ambas as classes.")
    return X_train, X_val, X_test, y_train, y_val, y_test
