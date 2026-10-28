"""Seleção de modelo e limiar por predições fora do fold no conjunto de treino.

Adaptação da referência: reaproveita cada ajuste para métricas e probabilidades
OOF, evitando treinar os mesmos folds duas vezes. O teste não é recebido aqui.
"""

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from module_olist.config import REPORTS_DIR
from module_olist.modeling.evaluate import calculate_metrics
from module_olist.pipeline import (
    create_gradient_boosting_pipeline, create_xgboost_pipeline,
    create_lightgbm_pipeline,
)

N_SPLITS = 5
RANDOM_STATE = 42


def find_best_threshold(y_true, y_proba):
    """Maximiza F1 OOF entre 0.01 e 0.99; empates favorecem o menor limiar.

    As métricas otimizadas são critérios de seleção, não estimativas
    independentes do desempenho do limiar escolhido.
    """
    rows = []
    for threshold in np.arange(1, 100) / 100:
        prediction = (y_proba >= threshold).astype(int)
        # As métricas de ranking não variam com o limiar; não são recalculadas.
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(y_true, prediction, zero_division=0),
            "recall": recall_score(y_true, prediction, zero_division=0),
            "f1": f1_score(y_true, prediction, zero_division=0),
        })
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["f1"].idxmax()].to_dict()
    return best, curve


def cross_validate_models(X_train, y_train):
    """Compara três modelos em cinco folds e retorna nome e limiar escolhidos.

    Medianas e categorias são ajustadas de novo dentro de cada fold.
    As médias por fold abaixo usam limiar 0.50; a seleção usa F1 OOF otimizado.
    """
    if y_train.isna().any() or set(y_train.unique()) != {0, 1}:
        raise ValueError("A validação exige as classes 0 e 1, sem ausentes.")
    if y_train.value_counts().min() < N_SPLITS:
        raise ValueError(f"São necessários ao menos {N_SPLITS} exemplos de cada classe no treino.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train e y_train devem ter índices alinhados.")

    pipelines = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X_train, y_train))
    model_rows, fold_rows, threshold_frames = [], [], []

    for name, pipeline in pipelines.items():
        logger.info(f"Validação cruzada: {name} | {N_SPLITS} folds")
        oof = np.full(len(y_train), np.nan)
        assignments = np.zeros(len(y_train), dtype=int)
        validation_scores, training_scores = [], []
        for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
            logger.info(f"[{name}] Treinando fold {fold}/{N_SPLITS}...")
            estimator = clone(pipeline)
            X_fit, y_fit = X_train.iloc[fit_idx], y_train.iloc[fit_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            estimator.fit(X_fit, y_fit)
            positive = list(estimator.classes_).index(1)
            fit_proba = estimator.predict_proba(X_fit)[:, positive]
            val_proba = estimator.predict_proba(X_val)[:, positive]
            oof[val_idx] = val_proba
            assignments[val_idx] += 1
            train_metrics = calculate_metrics(y_fit, (fit_proba >= 0.50).astype(int), fit_proba)
            val_metrics = calculate_metrics(y_val, (val_proba >= 0.50).astype(int), val_proba)
            training_scores.append(train_metrics)
            validation_scores.append(val_metrics)
            fold_rows.append({
                "model": name, "fold": fold, "n_fit": len(fit_idx), "n_validation": len(val_idx),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"validation_{k}": v for k, v in val_metrics.items()},
            })
            logger.info(f"[{name}] Fold {fold}: AP={val_metrics['average_precision']:.4f}")

        if not np.isfinite(oof).all() or not (assignments == 1).all():
            raise RuntimeError("Cada pedido precisa receber exatamente uma probabilidade OOF válida.")
        chosen, curve = find_best_threshold(y_train, oof)
        curve.insert(0, "model", name)
        threshold_frames.append(curve)
        row = {
            "model": name, "threshold": chosen["threshold"],
            "f1_oof_selected": chosen["f1"],
            "precision_oof_selected": chosen["precision"],
            "recall_oof_selected": chosen["recall"],
        }
        for metric in validation_scores[0]:
            scores = [v[metric] for v in validation_scores]
            row[f"{metric}_mean_05"] = float(np.mean(scores))
            row[f"{metric}_std_05"] = float(np.std(scores))
            row[f"{metric}_train_mean_05"] = float(np.mean([v[metric] for v in training_scores]))
        row["average_precision_gap"] = (
            row["average_precision_train_mean_05"] - row["average_precision_mean_05"]
        )
        model_rows.append(row)
        logger.success(f"[{name}] Limiar={chosen['threshold']:.2f} | F1 OOF de seleção={chosen['f1']:.4f}")

    # Ordenação estável preserva a ordem dos modelos em caso de empate.
    summary = pd.DataFrame(model_rows).sort_values(
        "f1_oof_selected", ascending=False, kind="stable",
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(REPORTS_DIR / "cv_folds.csv", index=False)
    pd.concat(threshold_frames, ignore_index=True).to_csv(REPORTS_DIR / "cv_thresholds.csv", index=False)
    winner = summary.iloc[0]
    logger.info("Comparação na validação:\n" + summary[[
        "model", "threshold", "f1_oof_selected", "average_precision_mean_05",
        "average_precision_std_05", "average_precision_gap",
    ]].round(4).to_string(index=False))
    logger.success(f"Selecionado: {winner['model']} | limiar: {winner['threshold']:.2f}")
    return str(winner["model"]), float(winner["threshold"])
