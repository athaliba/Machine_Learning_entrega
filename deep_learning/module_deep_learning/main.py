"""Treinamento da MLP com dados tabulares da Olist — Aula 14."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras

from module_deep_learning.config import EPOCHS, MODELS_DIR, OLIST_DATA, REPORTS_DIR, SEED, THRESHOLD
from module_deep_learning.dataset import load_dataset
from module_deep_learning.features import FEATURES, create_preprocessor, transform_features
from module_deep_learning.modeling.evaluate import evaluate_model
from module_deep_learning.modeling.train import train_model
from module_deep_learning.plots import generate_plots
from module_deep_learning.split import split_data


def run_training(dataset_path=OLIST_DATA, epochs=EPOCHS):
    if epochs < 1:
        raise ValueError("O número de épocas precisa ser positivo.")
    data = load_dataset(dataset_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    logger.info(f"Treino: {len(X_train):,} | Validação: {len(X_val):,} | Teste: {len(X_test):,}")
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    train_values = transform_features(preprocessor, X_train)
    val_values = transform_features(preprocessor, X_val)
    test_values = transform_features(preprocessor, X_test)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "mlp_preprocessor.joblib")
    model, history, class_weight, best_epoch = train_model(
        train_values, y_train, val_values, y_val, epochs=epochs,
    )
    report, prediction = evaluate_model(model, test_values, y_test, y_train)
    generate_plots(history, y_test, prediction)
    metadata = {
        "model": "MLP Keras", "model_file": "mlp_olist.keras",
        "preprocessor_file": "mlp_preprocessor.joblib", "features": FEATURES,
        "hidden_units": [128, 64], "activation": "relu", "output_activation": "sigmoid",
        "dropout": 0.3, "batch_normalization": True, "optimizer": "Adam",
        "learning_rate": 0.001, "batch_size": 256, "random_state": SEED,
        "epochs_requested": epochs, "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch, "selection_monitor": "val_pr_auc (Keras AUC PR aproximada)",
        "class_weight": class_weight, "threshold": THRESHOLD,
        "threshold_method": "fixo; nao otimizado com o teste",
        "n_train": len(X_train), "n_validation": len(X_val), "n_test": len(X_test),
        "dataset_file": Path(dataset_path).name,
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "python_version": platform.python_version(), "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__, "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "test_previously_used_in_olist": True,
    }
    (MODELS_DIR / "mlp_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.concat([
        pd.DataFrame({"order_id": data.loc[X.index, "order_id"], "split": name})
        for name, X in [("train", X_train), ("validation", X_val), ("test", X_test)]
    ], ignore_index=True).to_csv(REPORTS_DIR / "mlp_splits.csv", index=False)
    logger.warning(
        "O teste já foi utilizado no projeto Olist. Esta é uma avaliação descritiva, "
        "não uma amostra final inédita. Não ajuste a rede ou o limiar usando esse resultado."
    )
    logger.success("MLP treinada: modelo, pré-processador, metadados, relatórios e gráficos salvos.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Treina a MLP para atraso na Olist")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--data", type=Path, default=OLIST_DATA)
    args = parser.parse_args()
    run_training(args.data, args.epochs)


if __name__ == "__main__":
    main()
