"""Pré-processamento da MLP, ajustado exclusivamente no treino."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "promised_days", "item_count", "seller_count", "total_price", "total_freight",
    "approval_hours", "avg_item_price", "freight_ratio", "is_short_deadline",
    "is_weekend_purchase",
]
CATEGORICAL_FEATURES = ["purchase_hour", "purchase_weekday", "purchase_month", "customer_state"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_late"


def normalize_inputs(data):
    result = data[FEATURES].copy()
    for name in NUMERIC_FEATURES:
        result[name] = result[name].astype(float)
    for name in CATEGORICAL_FEATURES:
        values = result[name].astype("string")
        result[name] = values.astype(object).where(values.notna(), np.nan)
    return result


def create_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido",
                                  keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return Pipeline([
        ("types", FunctionTransformer(normalize_inputs, validate=False, feature_names_out="one-to-one")),
        ("columns", ColumnTransformer([
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ], remainder="drop")),
    ])


def transform_features(preprocessor, X):
    values = np.asarray(preprocessor.transform(X), dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError("O pré-processamento gerou valores não finitos.")
    return values
