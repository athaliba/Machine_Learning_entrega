"""Carregamento do pipeline e previsões com o limiar salvo na validação."""

import json
from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd

from module_olist.features import FEATURES


def validate_features(X, features):
    missing = [column for column in features if column not in X.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para prever: {missing}")
    return X[features].copy()


def validate_threshold(threshold):
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("O limiar deve estar entre 0 e 1, sem incluir os extremos.")
    return threshold


def load_model(model_path, metadata_path):
    """Carrega o arquivo confiável produzido pelo treinamento e confere as features."""
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Não encontrei o modelo da validação e seus metadados. "
            f"Confira {model_path} e {metadata_path}."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("features") != FEATURES:
        raise ValueError("As features atuais diferem das registradas no treinamento.")
    if metadata.get("model_file") != model_path.name:
        raise ValueError("Os metadados não correspondem ao nome do arquivo do modelo.")
    threshold = validate_threshold(metadata["threshold"])
    model = joblib.load(model_path)
    if list(model.feature_names_in_) != metadata["features"]:
        raise ValueError("As colunas do pipeline diferem dos metadados.")
    logger.info(f"Modelo carregado: {metadata['model_name']} | limiar: {threshold:.2f}")
    return model, metadata["model_name"], threshold


def classify_risk(y_proba, threshold):
    """Faixas heurísticas da referência, sem calibração ou garantia operacional.

    alto: escore >= 2 x limiar; médio: escore >= limiar; baixo: demais.
    Se o limiar for maior que 0.50, a faixa alto não será atingida.
    """
    threshold = validate_threshold(threshold)
    return np.select(
        [y_proba >= 2 * threshold, y_proba >= threshold],
        ["alto", "médio"], default="baixo",
    )


def predict(model, X, threshold):
    """Usa apenas predict_proba; nenhum componente é reajustado na inferência."""
    threshold = validate_threshold(threshold)
    if X.empty:
        return pd.DataFrame({
            "prob_is_late": pd.Series(dtype=float),
            "prediction": pd.Series(dtype="int8"),
            "risk_level": pd.Series(dtype=object),
        }, index=X.index)
    positive = list(model.classes_).index(1)
    y_proba = model.predict_proba(X)[:, positive]
    if not np.isfinite(y_proba).all() or ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("O modelo retornou escores inválidos.")
    return pd.DataFrame({
        "prob_is_late": y_proba,
        "prediction": (y_proba >= threshold).astype("int8"),
        "risk_level": classify_risk(y_proba, threshold),
    }, index=X.index)
