"""Simulação de active learning adaptada da referência da Aula 12.

Oculta rótulos já conhecidos para comparar estratégias com o mesmo orçamento.
Não altera o modelo salvo, o limiar ou as previsões das etapas anteriores.
"""

from pathlib import Path
import json

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import lightgbm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from module_olist.config import FIGURES_DIR, INTERIM_DATA_DIR, REPORTS_DIR
from module_olist.pipeline import create_lightgbm_pipeline
from module_olist.split import split_data


# ---------------------------------------------------------------------------
# Medida de confiança e estratégias de seleção
# ---------------------------------------------------------------------------


def max_predicted_probability(model, X_pool: pd.DataFrame) -> np.ndarray:
    """
    Maximum Predicted Probability (MPP): a maior probabilidade entre as classes.
    Perto de 1.0 = modelo confiante. Perto de 0.5 = modelo em dúvida.
    """
    proba = model.predict_proba(X_pool)
    return proba.max(axis=1)


def uncertainty_sampling(model, X_pool: pd.DataFrame, n_queries: int, rng=None) -> np.ndarray:
    """Seleciona os n_queries exemplos com MENOR confiança (Top-k mais incertos)."""
    confidence = max_predicted_probability(model, X_pool)
    return np.argsort(confidence, kind="stable")[:min(n_queries, len(X_pool))]


def random_sampling(model, X_pool: pd.DataFrame, n_queries: int, rng=None) -> np.ndarray:
    """Baseline: escolhe exemplos do pool ao acaso."""
    rng = rng if rng is not None else np.random.default_rng(42)
    return rng.choice(len(X_pool), size=min(n_queries, len(X_pool)), replace=False)


STRATEGIES = {
    "incerteza": uncertainty_sampling,
    "aleatoria": random_sampling,
}


# ---------------------------------------------------------------------------
# Loop de active learning
# ---------------------------------------------------------------------------


def run_active_learning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy: str = "incerteza",
    initial_fraction: float = 0.10,
    n_queries: int = 500,
    n_rounds: int = 10,
    random_state: int = 42,
    query_log: list | None = None,
) -> pd.DataFrame:
    """
    Simula o active learning e devolve, por rodada, quantos rótulos foram
    usados e as métricas no conjunto de teste.

    - initial_fraction: fração do treino que já começa rotulada (10%, como na aula)
    - n_queries: quantos pedidos o "humano" rotula por rodada
    - n_rounds: quantas rodadas de consulta
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    if not 0 < initial_fraction < 1 or n_queries < 1 or n_rounds < 0:
        raise ValueError("Use fração inicial entre 0 e 1, consultas positivas e rodadas não negativas.")
    if not X_train.index.equals(y_train.index) or not X_test.index.equals(y_test.index):
        raise ValueError("As features e os alvos devem ter índices alinhados.")
    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        raise ValueError("Treino e avaliação devem conter ambas as classes.")
    if y_train.value_counts().min() < 2:
        raise ValueError("A inicialização estratificada exige ao menos dois exemplos por classe.")
    rng = np.random.default_rng(random_state)
    original_index = X_train.index.to_numpy()
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    n_initial = int(len(X_train) * initial_fraction)
    if n_initial < 2 or len(X_train) - n_initial < 2:
        raise ValueError("A fração inicial precisa reservar ao menos dois pedidos para cada parte.")
    # Inicialização estratificada compartilhada pelas estratégias. A simulação
    # usa rótulos conhecidos para montá-la; um processo real precisaria obter
    # esse conjunto inicial antes de iniciar as consultas.
    labeled, unlabeled = train_test_split(
        np.arange(len(X_train)), train_size=n_initial,
        stratify=y_train, random_state=random_state,
    )
    labeled, unlabeled = np.sort(labeled), np.sort(unlabeled)
    if y_train.iloc[labeled].nunique() != 2:
        raise ValueError("O conjunto inicial ficou sem uma classe. Aumente a fração inicial.")
    if query_log is not None:
        query_log.extend({
            "estrategia": strategy, "rodada_inclusao": 0,
            "indice_base": int(original_index[i]), "rotulo_revelado": int(y_train.iloc[i]),
        } for i in labeled)

    query_fn = STRATEGIES[strategy]
    history = []

    for round_ in range(n_rounds + 1):
        # 1. Treina só com o que já tem rótulo
        model = create_lightgbm_pipeline()
        model.fit(X_train.iloc[labeled], y_train.iloc[labeled])

        # 2. Avalia no teste (Average Precision é a métrica principal: classes desbalanceadas)
        proba_test = model.predict_proba(X_test)[:, 1]
        history.append(
            {
                "estrategia": strategy,
                "rodada": round_,
                "n_rotulados": len(labeled),
                "fracao_rotulada": len(labeled) / len(X_train),
                "n_positivos_rotulados": int(y_train.iloc[labeled].sum()),
                "roc_auc": roc_auc_score(y_test, proba_test),
                "average_precision": average_precision_score(y_test, proba_test),
            }
        )
        logger.info(
            f"[{strategy}] rodada {round_}: {len(labeled)} rótulos | "
            f"ROC-AUC={history[-1]['roc_auc']:.3f} | Average Precision={history[-1]['average_precision']:.3f}"
        )

        if round_ == n_rounds or len(unlabeled) == 0:
            break

        # 3. Modelo escolhe quais pedidos quer que o humano rotule
        chosen = query_fn(model, X_train.iloc[unlabeled], n_queries, rng)
        new_idx = unlabeled[chosen]
        if len(np.unique(new_idx)) != len(new_idx):
            raise RuntimeError("Uma consulta repetiu pedidos do pool.")
        if query_log is not None:
            query_log.extend({
                "estrategia": strategy, "rodada_inclusao": round_ + 1,
                "indice_base": int(original_index[i]), "rotulo_revelado": int(y_train.iloc[i]),
            } for i in new_idx)

        # 4. Human-in-the-loop: o "oráculo" revela y (aqui simulado com o y real)
        labeled = np.concatenate([labeled, new_idx])
        unlabeled = np.setdiff1d(unlabeled, new_idx)

    return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Comparação e relatório
# ---------------------------------------------------------------------------


def full_label_reference(X_train, y_train, X_test, y_test) -> float:
    """
    Referência: Average Precision do mesmo modelo treinado com TODOS os rótulos.
    Mostra o quanto o active learning chega perto do cenário de referência
    usando só uma fração dos rótulos.
    """
    model = create_lightgbm_pipeline()
    model.fit(X_train, y_train)
    return average_precision_score(y_test, model.predict_proba(X_test)[:, 1])


def plot_learning_curves(
    results: pd.DataFrame,
    output_path: Path,
    reference_ap: float | None = None,
    n_total: int | None = None,
) -> None:
    """Plota Average Precision x número de pedidos rotulados para cada estratégia."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, df in results.groupby("estrategia"):
        ax.plot(df["n_rotulados"], df["average_precision"], marker="o", label=strategy)

    if reference_ap is not None:
        label = "todos os rótulos"
        if n_total is not None:
            label += f" ({n_total:,} pedidos)".replace(",", ".")
        ax.axhline(reference_ap, color="gray", linestyle="--", label=label)

    ax.set_xlabel("Pedidos rotulados")
    ax.set_ylabel("Average Precision no teste")
    ax.set_title("Active learning: incerteza x amostragem aleatória")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info(f"Curva salva em: {output_path}")


def run_experiment(initial_fraction=0.10, n_queries=500, n_rounds=10, random_state=42):
    logger.info("Iniciando simulação de active learning...")

    data = pd.read_csv(INTERIM_DATA_DIR / "orders_dataset_refined.csv")
    X_train, X_test, y_train, y_test = split_data(data)

    query_log = []
    results = pd.concat(
        [
            run_active_learning(
                X_train, y_train, X_test, y_test, strategy=s,
                initial_fraction=initial_fraction, n_queries=n_queries,
                n_rounds=n_rounds, random_state=random_state, query_log=query_log,
            )
            for s in STRATEGIES
        ],
        ignore_index=True,
    )

    reference = full_label_reference(X_train, y_train, X_test, y_test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "active_learning_results.csv", index=False)
    queries = pd.DataFrame(query_log)
    queries["order_id"] = queries["indice_base"].map(data["order_id"])
    queries.to_csv(REPORTS_DIR / "active_learning_queries.csv", index=False)
    plot_learning_curves(
        results,
        FIGURES_DIR / "active_learning_curve.png",
        reference_ap=reference,
        n_total=len(X_train),
    )

    final = results.groupby("estrategia").tail(1)
    logger.success(f"Resultado final:\n{final.to_string(index=False)}")

    used = final["n_rotulados"].iloc[0] / len(X_train)
    summary = {
        "simulation": True, "model": "LightGBM",
        "initial_fraction": initial_fraction, "n_queries": n_queries,
        "n_rounds": n_rounds, "random_state": random_state,
        "n_train": len(X_train), "n_test": len(X_test),
        "reference_average_precision_all_labels": float(reference),
        "test_previously_evaluated": True,
        "evaluation_used_for_queries": False,
        "n_seeds": 1,
        "sklearn_version": sklearn.__version__, "lightgbm_version": lightgbm.__version__,
        "final_results": final.to_dict(orient="records"),
    }
    (REPORTS_DIR / "active_learning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(
        f"Referência com todos os {len(X_train):,} rótulos: Average Precision={reference:.3f}. "
        f"O active learning usou {used:.0%} dos rótulos."
    )

    logger.warning(
        "Simulação com uma semente; a seleção por incerteza pode não superar a aleatória. "
        "As métricas por rodada são descritivas; não use o teste para ajustar o experimento."
    )
    return results


def main():
    run_experiment()


if __name__ == "__main__":
    main()
