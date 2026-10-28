"""Tree SHAP com o mesmo pré-processamento do modelo treinado."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
import shap


def prepare_data_for_shap(pipeline, X):
    
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    return pd.DataFrame(
        transformed, columns=preprocessor.get_feature_names_out(), index=X.index,
    )


def create_explainer(pipeline):
    """Explica a saída bruta dos classificadores binários de boosting."""
    return shap.TreeExplainer(
        pipeline.named_steps["model"], model_output="raw",
        feature_perturbation="tree_path_dependent",
    )


def calculate_shap_values(explainer, X_transformed):
    explanation = explainer(X_transformed, check_additivity=True)
    # Algumas versões de explicadores retornam também uma dimensão de classe.
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    if explanation.values.ndim != 2:
        raise ValueError("Formato SHAP inesperado para classificação binária.")
    return explanation


def map_to_original_feature(transformed_name, categorical_features):
    name = transformed_name.split("__", 1)[-1]
    for feature in sorted(categorical_features, key=len, reverse=True):
        if name.startswith(feature + "_"):
            return feature
    return name


def aggregate_shap_by_feature(shap_values, X_original, categorical_features):
    """Soma contribuições das colunas One-Hot de cada feature original.

    É uma agregação da explicação das colunas codificadas, não um novo cálculo
    de Shapley com grupos de variáveis. Preserva a soma de cada observação.
    """
    names = [map_to_original_feature(n, categorical_features) for n in shap_values.feature_names]
    values = pd.DataFrame(shap_values.values, columns=names)
    grouped = values.T.groupby(level=0, sort=False).sum().T
    original = X_original[grouped.columns].reset_index(drop=True)
    display = original.astype(object).where(original.notna(), "ausente (imputado)")
    numeric = original.apply(pd.to_numeric, errors="coerce")
    result = shap.Explanation(
        values=grouped.to_numpy(dtype=float),
        base_values=np.asarray(shap_values.base_values),
        data=numeric.to_numpy(dtype=float, na_value=np.nan),
        display_data=display.to_numpy(dtype=object),
        feature_names=list(grouped.columns),
    )
    np.testing.assert_allclose(
        result.values.sum(axis=1),
        shap_values.values.sum(axis=1),
        rtol=1e-5,
        atol=1e-6,
    )
    return result


def check_probability_reconstruction(pipeline, X, explanation):
    """Confere se base + contribuições reconstrói a previsão em log-odds."""
    total = np.asarray(explanation.base_values).reshape(-1) + explanation.values.sum(axis=1)
    positive = list(pipeline.classes_).index(1)
    proba = pipeline.predict_proba(X)[:, positive]
    np.testing.assert_allclose(expit(total), proba, rtol=3e-4, atol=3e-5)
    return float(np.max(np.abs(expit(total) - proba)))
