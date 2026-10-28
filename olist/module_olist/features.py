"""Variáveis disponíveis no instante da aprovação do pagamento."""

import pandas as pd

NUMERIC_FEATURES = [
    "promised_days", "item_count", "seller_count", "total_price",
    "total_freight", "approval_hours", "avg_item_price", "freight_ratio",
    "is_short_deadline", "is_weekend_purchase",
]
CATEGORICAL_FEATURES = [
    "purchase_hour", "purchase_weekday", "purchase_month", "customer_state",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_late"
SHORT_DEADLINE_DAYS = 7


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos sem excluir pedidos ou aprender parâmetros do histórico.

    Valores inconsistentes nos cálculos ficam ausentes para inspeção.
    A imputação será definida na etapa de pipeline, somente com o treino.
    A função também aceita uma base sem alvo.
    """
    result = data.copy()
    purchase = result["order_purchase_timestamp"]
    approved = result["order_approved_at"]
    deadline = result["order_estimated_delivery_date"]

    promised = (deadline - approved).dt.total_seconds().div(86400)
    result["promised_days"] = promised.where(promised.gt(0))
    approval = (approved - purchase).dt.total_seconds().div(3600)
    result["approval_hours"] = approval.where(approval.ge(0))

    result["purchase_month"] = purchase.dt.month.astype("Int64")
    result["purchase_weekday"] = purchase.dt.dayofweek.astype("Int64")
    result["purchase_hour"] = purchase.dt.hour.astype("Int64")
    result["is_weekend_purchase"] = result["purchase_weekday"].ge(5).astype("Int8")

    # Sete dias é uma hipótese de trabalho; sua utilidade será avaliada depois.
    result["is_short_deadline"] = (
        result["promised_days"].le(SHORT_DEADLINE_DAYS).astype("Int8")
        .mask(result["promised_days"].isna())
    )
    quantity = result["item_count"].where(result["item_count"].gt(0))
    price = result["total_price"].where(result["total_price"].gt(0))
    result["avg_item_price"] = result["total_price"].div(quantity)
    result["freight_ratio"] = result["total_freight"].div(price)
    return result
