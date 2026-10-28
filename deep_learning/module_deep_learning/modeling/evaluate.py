"""Métricas no teste para a rede e para uma referência da classe majoritária."""

import json

from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
import numpy as np

from module_deep_learning.config import REPORTS_DIR, THRESHOLD
from module_deep_learning.modeling.train import make_dataset


def calculate_metrics(y_true, proba, threshold=THRESHOLD):
    prediction = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
    }


def evaluate_model(model, X_test, y_test, y_train):
    proba = model.predict(make_dataset(X_test), verbose=0).ravel()
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise ValueError("A rede gerou escores inválidos.")
    prediction = (proba >= THRESHOLD).astype(int)
    metrics = calculate_metrics(y_test, proba)
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    baseline = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, list(dummy.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    report = {
        "model": "MLP Keras (128, 64)", "threshold": THRESHOLD, "n_test": len(y_test),
        "metrics": metrics, "positive_rate_test": float(np.mean(y_test)),
        "baseline_most_frequent": calculate_metrics(y_test, baseline),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_previously_used_in_olist": True,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "mlp_test_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for name, value in metrics.items():
        logger.info(f"Teste — {name}: {value:.4f}")
    logger.info(f"Atrasos encontrados: {tp:,} de {tp + fn:,} | Alertas falsos: {fp:,}")
    return report, prediction
