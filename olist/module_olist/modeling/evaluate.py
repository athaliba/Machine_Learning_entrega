"""Avaliação do holdout e comparação com a classe majoritária."""

import json
from pathlib import Path

from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from module_olist.config import FIGURES_DIR, REPORTS_DIR


def calculate_metrics(y_true, y_pred, y_proba):
    """AP e ROC-AUC usam escores; as demais métricas usam as classes previstas."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }


def evaluate_baseline(X_train, y_train, X_test, y_test):
    baseline = DummyClassifier(strategy="most_frequent")
    # O baseline ignora os atributos; entradas constantes dispensam tratamento.
    baseline.fit(np.zeros((len(X_train), 1)), y_train)
    dummy_test = np.zeros((len(X_test), 1))
    positive_index = list(baseline.classes_).index(1)
    return calculate_metrics(
        y_test, baseline.predict(dummy_test),
        baseline.predict_proba(dummy_test)[:, positive_index],
    )


def plot_confusion_matrix(y_true, y_pred, model_name, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=["No prazo", "Atrasado"],
        cmap="Blues", values_format=",d", ax=ax,
    )
    ax.set(title=f"Matriz de confusão — {model_name} (teste)",
           xlabel="Previsto", ylabel="Real")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Matriz de confusão salva em: {output_path}")


def evaluate_model(
    model, model_name, X_test, y_test, threshold,
    X_train=None, y_train=None,
    metrics_path=REPORTS_DIR / "test_metrics.json",
    figure_path=FIGURES_DIR / "confusion_matrix.png",
):
    """Avalia a configuração fixa; não seleciona modelo ou limiar pelo teste."""
    positive_index = list(model.classes_).index(1)
    y_proba = model.predict_proba(X_test)[:, positive_index]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    logger.success(f"Avaliação: {model_name} | limiar: {threshold:.2f}")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
    logger.info(f"Atrasos encontrados: {tp:,} de {tp + fn:,} | Alertas falsos: {fp:,}")

    report = {
        "model_name": model_name, "threshold": float(threshold),
        "n_test": len(y_test), "positive_rate_test": float(y_test.mean()),
        "validation": "holdout aleatorio estratificado por pedido",
        "metrics": {key: float(value) for key, value in metrics.items()},
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if X_train is not None and y_train is not None:
        baseline = evaluate_baseline(X_train, y_train, X_test, y_test)
        report["baseline_most_frequent"] = {
            key: float(value) for key, value in baseline.items()
        }
        logger.info("Referência da classe majoritária: " + json.dumps(baseline))
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    plot_confusion_matrix(y_test, y_pred, model_name, figure_path)
    logger.success(f"Métricas salvas em: {metrics_path}")
    return report
