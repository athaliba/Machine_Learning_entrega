"""Leitura da base refinada já produzida pelo projeto Olist."""

from pathlib import Path

from loguru import logger
import pandas as pd

from module_deep_learning.config import OLIST_DATA
from module_deep_learning.features import FEATURES, TARGET


def load_dataset(dataset_path=OLIST_DATA):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Não encontrei a base refinada da Olist: {dataset_path}")
    data = pd.read_csv(dataset_path)
    missing = sorted(set(["order_id", *FEATURES, TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"A base está sem colunas necessárias: {missing}")
    if data["order_id"].isna().any() or not data["order_id"].is_unique:
        raise ValueError("A base deve ter um identificador válido por pedido.")
    if data[TARGET].isna().any() or set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("O alvo deve conter 0 e 1, sem ausentes.")
    logger.info(f"Pedidos: {len(data):,} | Taxa de atraso: {data[TARGET].mean():.2%}")
    return data
