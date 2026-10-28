"""Treinamento e persistência do modelo inicial."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
from loguru import logger
import pandas as pd
import sklearn

from module_olist.pipeline import create_gradient_boosting_pipeline


def create_selected_model(model_name):
    if model_name != "Gradient Boosting":
        raise ValueError(f"Modelo não implementado nesta etapa: {model_name}")
    return create_gradient_boosting_pipeline()


def train_model(model_name, threshold, X_train, y_train, model_path, metadata_path):
    """Ajusta o pipeline somente no treino e salva modelo e metadados."""
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    logger.info(f"Treinando: {model_name} | limiar fixo: {threshold:.2f}")
    model = create_selected_model(model_name)
    model.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "threshold": float(threshold),
        "selection_method": "modelo fixado antes da avaliação do teste",
        "threshold_method": "fixo em 0.50; sem otimização nesta etapa",
        "validation": "holdout aleatorio estratificado por pedido",
        "test_size": 0.2, "random_state": 42,
        "features": list(X_train.columns),
        "target": "is_late",
        "target_rule": "delivered_customer_timestamp > estimated_delivery_timestamp",
        "n_train": len(X_train),
        "positive_rate_train": float(y_train.mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "model_parameters": model.named_steps["model"].get_params(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    logger.success(f"Pipeline salvo em: {model_path}")
    return model
