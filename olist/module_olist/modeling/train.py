"""Reajuste do modelo selecionado na validação e persistência."""

from datetime import datetime, timezone
import json
import importlib
from pathlib import Path
import platform

import joblib
from loguru import logger
import pandas as pd
import sklearn

from module_olist.pipeline import (
    create_gradient_boosting_pipeline, create_xgboost_pipeline,
    create_lightgbm_pipeline,
)


def create_selected_model(model_name):
    factories = {
        "Gradient Boosting": create_gradient_boosting_pipeline,
        "XGBoost": create_xgboost_pipeline,
        "LightGBM": create_lightgbm_pipeline,
    }
    if model_name not in factories:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    return factories[model_name]()


def train_model(model_name, threshold, X_train, y_train, model_path, metadata_path):
    """Ajusta o pipeline somente no treino e salva modelo e metadados."""
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    logger.info(f"Treinando: {model_name} | limiar escolhido no treino: {threshold:.2f}")
    model = create_selected_model(model_name)
    model.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "threshold": float(threshold),
        "selection_method": "maior F1 OOF com limiar otimizado no conjunto de treino",
        "threshold_method": "maximiza F1 OOF na grade 0.01 a 0.99, passo 0.01",
        "validation": "StratifiedKFold em 5 folds sobre os 80% de treino",
        "cv_n_splits": 5,
        "selection_scores_are_independent_estimates": False,
        "test_previously_evaluated_in_aula07": True,
        "model_file": model_path.name,
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
        "booster_versions": {
            name: importlib.import_module(name).__version__
            for name in ["xgboost", "lightgbm"]
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    logger.success(f"Pipeline salvo em: {model_path}")
    return model
