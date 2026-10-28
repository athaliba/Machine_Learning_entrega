"""Aplica o modelo salvo aos pedidos abertos observados no CSV histórico."""

import json
from pathlib import Path

from loguru import logger
import pandas as pd

from module_olist.config import MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR
from module_olist.dataset import OPEN_STATUSES, create_inference_dataset, load_dataset
from module_olist.features import FEATURES, create_features
from module_olist.modeling.predict import load_model, predict, validate_features

ID_COLUMNS = ["order_id", "order_status", "order_approved_at", "customer_state"]


def load_open_orders():
    orders, items, customers = load_dataset(
        RAW_DATA_DIR / "olist_orders_dataset.csv",
        RAW_DATA_DIR / "olist_order_items_dataset.csv",
        RAW_DATA_DIR / "olist_customers_dataset.csv",
    )
    data = create_inference_dataset(orders, items, customers)
    open_count = int(orders["order_status"].isin(OPEN_STATUSES).sum())
    logger.info(f"Pedidos com status aberto: {open_count:,} | Elegíveis: {len(data):,}")
    logger.info(
        f"Excluídos por datas necessárias ausentes ou entrega já registrada: {open_count - len(data):,}"
    )
    return create_features(data)


def run_inference(data=None, model=None, model_name=None, threshold=None, output_path=None):
    """Gera o ranking de pedidos sem criar alvo ou calcular métricas de acerto."""
    if data is None:
        data = load_open_orders()
    if not data.index.is_unique or not data["order_id"].is_unique:
        raise ValueError("A inferência exige um índice único e uma linha por pedido.")
    if model is None:
        model, model_name, threshold = load_model(
            MODELS_DIR / "best_model.joblib", MODELS_DIR / "metadata_cv.json",
        )
    if threshold is None:
        raise ValueError("É necessário informar o limiar associado ao modelo.")
    X = validate_features(data, FEATURES)
    predictions = predict(model, X, threshold)
    result = (
        data[ID_COLUMNS]
        .join(X[["promised_days", "total_price", "total_freight"]])
        .join(predictions)
        .sort_values("prob_is_late", ascending=False, kind="stable")
    )
    result["missing_item_data"] = data["item_count"].isna().reindex(result.index)
    result["missing_customer_state"] = data["customer_state"].isna().reindex(result.index)
    output_path = Path(output_path) if output_path is not None else MODELS_DIR / "open_orders_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    counts = result["risk_level"].value_counts().reindex(["alto", "médio", "baixo"], fill_value=0)
    summary = {
        "model_name": model_name, "threshold": float(threshold),
        "prediction_file": output_path.name,
        "n_orders": len(result), "n_flagged": int(result["prediction"].sum()),
        "risk_counts": {name: int(value) for name, value in counts.items()},
        "missing_features_before_pipeline": {name: int(value) for name, value in X.isna().sum().items()},
        "scope": "pedidos abertos observados no CSV historico; sem simulacao temporal",
        "risk_bands": "heuristica da referencia; escores nao calibrados",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(f"Modelo: {model_name} | Limiar: {float(threshold):.2f} | Pedidos avaliados: {len(result):,}")
    logger.info("Pedidos por faixa heurística:\n" + counts.to_string())
    logger.info(
        "10 pedidos com maior escore:\n" + result.head(10)[[
            "order_id", "customer_state", "promised_days", "prob_is_late", "risk_level",
        ]].to_string(index=False)
    )
    if X.isna().any().any():
        logger.warning("Há features ausentes; o pipeline aplicou o tratamento aprendido no treino.")
    if result.empty:
        logger.info("Nenhum pedido elegível; foi salvo um CSV vazio com os cabeçalhos.")
    logger.success(f"Previsões salvas em: {output_path}")
    return result


def main():
    pd.set_option("display.width", 120)
    run_inference()


if __name__ == "__main__":
    main()
