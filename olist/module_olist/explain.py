"""Explicações globais e locais, adaptadas do projeto de referência."""

import json

from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from module_olist.config import FIGURES_DIR, INTERIM_DATA_DIR, MODELS_DIR, REPORTS_DIR
from module_olist.features import CATEGORICAL_FEATURES, FEATURES
from module_olist.modeling.interpret import (
    aggregate_shap_by_feature, calculate_shap_values, check_probability_reconstruction,
    create_explainer, prepare_data_for_shap,
)
from module_olist.modeling.predict import load_model, predict
from module_olist.split import split_data

MAX_DISPLAY = 10
LOCAL_COLUMNS = ["order_id", "customer_state", "prob_is_late", "risk_level", "principais_fatores"]


def explain_rows(model, X, explainer):
    transformed = prepare_data_for_shap(model, X)
    values = calculate_shap_values(explainer, transformed)
    result = aggregate_shap_by_feature(values, X, CATEGORICAL_FEATURES)
    error = check_probability_reconstruction(model, X, result)
    return result, error


def save_figure(path):
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close("all")


def top_factors(explanation, row, n=3):
    contributions = pd.Series(explanation.values[row], index=explanation.feature_names)
    observed = pd.Series(explanation.display_data[row], index=explanation.feature_names)
    positives = contributions[contributions > 0].sort_values(ascending=False).head(n)
    factors = []
    for name, contribution in positives.items():
        value = observed[name]
        formatted = f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
        factors.append(f"{name}={formatted} (+{contribution:.3f} log-odds)")
    return "; ".join(factors) or "Nenhuma contribuição positiva nesta explicação."


def explain_open_orders(model, threshold, explainer):
    """Explica pedidos altos após conferir que o ranking corresponde ao modelo."""
    path = MODELS_DIR / "open_orders_predictions.csv"
    output = REPORTS_DIR / "open_orders_explanations.csv"
    if not path.is_file():
        logger.warning("Ranking não encontrado; execute --mode infer para obter explicações de pedidos abertos.")
        pd.DataFrame(columns=LOCAL_COLUMNS).to_csv(output, index=False)
        return None
    from module_olist.inference import load_open_orders
    opened = load_open_orders()
    saved = pd.read_csv(path, dtype={"order_id": str})
    if not saved["order_id"].is_unique or set(saved["order_id"]) != set(opened["order_id"]):
        raise ValueError("O ranking difere dos pedidos atuais do CSV. Execute --mode infer novamente.")
    aligned = opened.set_index("order_id").loc[saved["order_id"]]
    fresh = predict(model, aligned[FEATURES], threshold)
    if not np.allclose(saved["prob_is_late"], fresh["prob_is_late"], rtol=1e-5, atol=1e-7):
        raise ValueError("As previsões salvas não correspondem ao modelo/dados atuais. Execute --mode infer.")
    if saved["risk_level"].tolist() != fresh["risk_level"].tolist():
        raise ValueError("As faixas salvas diferem do limiar atual. Execute --mode infer.")
    risky = saved.loc[saved["risk_level"].eq("alto")].sort_values(
        "prob_is_late", ascending=False, kind="stable",
    )
    if risky.empty:
        pd.DataFrame(columns=LOCAL_COLUMNS).to_csv(output, index=False)
        logger.info("Nenhum pedido na faixa alta; o waterfall usará uma observação da amostra global.")
        return None
    X = opened.set_index("order_id").loc[risky["order_id"], FEATURES]
    explanation, error = explain_rows(model, X, explainer)
    report = risky[["order_id", "customer_state", "prob_is_late", "risk_level"]].copy()
    report["principais_fatores"] = [top_factors(explanation, i) for i in range(len(report))]
    report.to_csv(output, index=False)
    logger.info("Exemplos de contribuições que elevam a saída do modelo:\n" + report.head(5).to_string(index=False))
    logger.success(f"Pedidos altos explicados: {len(report):,} | Arquivo: {output}")
    return explanation[0], str(report.iloc[0]["order_id"]), len(report), error


def generate_explanations(sample_size=1000):
    if sample_size < 1:
        raise ValueError("sample_size precisa ser positivo.")
    data_path = INTERIM_DATA_DIR / "orders_dataset_refined.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Base refinada não encontrada em {data_path}.")
    model, model_name, threshold = load_model(
        MODELS_DIR / "best_model.joblib", MODELS_DIR / "metadata_cv.json",
    )
    data = pd.read_csv(data_path)
    # Repete a separação original para escolher a amostra descritiva do treino.
    X_train, _, _, _ = split_data(data)
    X_sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Calculando SHAP: {model_name} | amostra de treino: {len(X_sample):,}")
    explainer = create_explainer(model)
    explanation, global_error = explain_rows(model, X_sample, explainer)
    importance = pd.DataFrame(
        np.abs(explanation.values), columns=explanation.feature_names,
    ).mean().sort_values(ascending=False).to_frame("mean_abs_shap")
    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index_label="feature")
    logger.info("Importância global na amostra (média |SHAP|):\n" + importance.round(4).to_string())

    shap.plots.bar(explanation, max_display=MAX_DISPLAY, show=False)
    plt.title(f"Importância global — {model_name}")
    plt.xlabel("Média |SHAP| (log-odds)")
    save_figure(FIGURES_DIR / "shap_global_bar.png")

    shap.plots.beeswarm(explanation, max_display=MAX_DISPLAY, show=False)
    plt.title("Contribuições na amostra de treino")
    plt.xlabel("SHAP (log-odds): negativo reduz a saída; positivo aumenta")
    save_figure(FIGURES_DIR / "shap_beeswarm.png")

    local = explain_open_orders(model, threshold, explainer)
    if local is None:
        row, order_id, n_open, local_error = explanation[0], None, 0, None
        title = "Explicação de um pedido da amostra de treino"
    else:
        row, order_id, n_open, local_error = local
        title = f"Pedido aberto {order_id[:8]}… — contribuições do modelo"
    # Preserva o valor textual do estado e identifica ausências no waterfall.
    waterfall_row = shap.Explanation(
        values=row.values, base_values=float(row.base_values),
        data=np.asarray(row.display_data, dtype=object), feature_names=row.feature_names,
    )
    shap.plots.waterfall(waterfall_row, max_display=MAX_DISPLAY, show=False)
    plt.title(title)
    plt.gcf().text(0.5, -0.02, "Saída em log-odds; contribuições não são causas do atraso.",
                  ha="center", fontsize=9)
    save_figure(FIGURES_DIR / "shap_waterfall.png")

    summary = {
        "model_name": model_name, "threshold": threshold,
        "sample_source": "particao de treino; amostra descritiva com semente 42",
        "n_global_sample": len(X_sample), "n_open_high_explained": n_open,
        "waterfall_order_id": order_id,
        "output_units": "log-odds (raw)", "shap_version": shap.__version__,
        "feature_perturbation": "tree_path_dependent",
        "max_probability_reconstruction_error_global": global_error,
        "max_probability_reconstruction_error_open": local_error,
        "feature_grouping": "soma das contribuicoes das colunas codificadas",
        "limitations": "explica o modelo, nao causalidade; escores nao calibrados",
    }
    (REPORTS_DIR / "shap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.success(f"Gráficos e relatórios SHAP salvos em {FIGURES_DIR} e {REPORTS_DIR}.")
    return summary


def main():
    generate_explanations()


if __name__ == "__main__":
    main()
