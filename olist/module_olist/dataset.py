"""Leitura e integração do histórico de pedidos da Olist."""

from pathlib import Path

import pandas as pd


DATE_COLUMNS = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def load_dataset(orders_path: Path, items_path: Path, customers_path: Path):
    """Lê as três tabelas, preservando datas e zeros no prefixo de CEP."""
    orders = pd.read_csv(orders_path, parse_dates=DATE_COLUMNS)
    items = pd.read_csv(items_path)
    customers = pd.read_csv(
        customers_path, dtype={"customer_zip_code_prefix": str}
    )
    return orders, items, customers


def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Seleciona entregas observadas e cria o alvo conforme a regra da EDA.

    A comparação usa timestamps completos. Uma entrega no dia prometido,
    após 00:00, pode ser classificada como atrasada. Essa convenção deve
    ser revista em conjunto com a EDA caso o prazo represente o dia inteiro.
    O recorte permite estudar atraso entre entregues, não cancelamentos.
    """
    required = ["order_delivered_customer_date",
                "order_estimated_delivery_date", "order_approved_at"]
    eligible = orders["order_status"].eq("delivered") & orders[required].notna().all(axis=1)
    data = orders.loc[eligible].copy()
    data["is_late"] = (
        data["order_delivered_customer_date"]
        > data["order_estimated_delivery_date"]
    ).astype("int8")
    return data


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """Resume itens e vendedores em uma linha por pedido."""
    return items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "size"),
        seller_count=("seller_id", "nunique"),
        total_price=("price", lambda values: values.sum(min_count=1)),
        total_freight=("freight_value", lambda values: values.sum(min_count=1)),
    )


def create_dataset(orders, items, customers) -> pd.DataFrame:
    """Integra o histórico, preservando pedidos sem correspondência para inspeção."""
    data = create_target(orders)
    data = data.merge(aggregate_items(items), on="order_id", how="left",
                      validate="one_to_one")
    return data.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id", how="left", validate="many_to_one",
    )


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Salva uma base sem adicionar o índice como coluna."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
