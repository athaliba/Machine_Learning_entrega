"""Pipelines dos três modelos comparados na validação cruzada."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from module_olist.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def normalize_inputs(data):
    """Uniformiza tipos e ausências sem aprender parâmetros do histórico."""
    result = data[FEATURES].copy()
    for column in NUMERIC_FEATURES:
        result[column] = result[column].astype(float)
    for column in CATEGORICAL_FEATURES:
        values = result[column].astype("string")
        result[column] = values.astype(object).where(values.notna(), np.nan)
    return result


def create_preprocessor():
    numeric = SimpleImputer(strategy="median", keep_empty_features=True)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido",
                                  keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")
    return Pipeline([
        ("types", FunctionTransformer(
            normalize_inputs, validate=False, feature_names_out="one-to-one",
        )),
        ("columns", columns),
    ])


def create_gradient_boosting_pipeline():
    """Mantém a configuração de Gradient Boosting presente na referência."""
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
        )),
    ])


def create_xgboost_pipeline():
    from xgboost import XGBClassifier

    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=42, n_jobs=2, eval_metric="logloss",
        )),
    ])


def create_lightgbm_pipeline():
    from lightgbm import LGBMClassifier

    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=42, n_jobs=2, verbosity=-1,
        )),
    ])
