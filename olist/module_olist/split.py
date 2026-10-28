"""Holdout inicial por pedido, conforme a organização da referência."""

from sklearn.model_selection import train_test_split

from module_olist.features import FEATURES, TARGET


def split_data(data):
    """Reserva 20% para teste, preservando aproximadamente a proporção do alvo.

    A divisão aleatória avalia o histórico; não simula períodos futuros.
    Pedidos de um mesmo cliente podem estar nos dois conjuntos.
    """
    if not data["order_id"].is_unique:
        raise ValueError("A base deve ter uma linha por pedido.")
    y = data[TARGET]
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("O alvo precisa conter 0 e 1, sem valores ausentes.")
    if y.value_counts().min() < 2:
        raise ValueError("São necessários ao menos dois pedidos de cada classe.")
    return train_test_split(
        data[FEATURES].copy(), y.copy(), test_size=0.2,
        stratify=y, random_state=42,
    )
