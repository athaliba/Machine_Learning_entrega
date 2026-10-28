"""Curvas de aprendizagem e matriz de confusão da MLP."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from module_deep_learning.config import FIGURES_DIR


def generate_plots(history, y_test, prediction):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="Treino (loss ponderada)")
    axes[0].plot(history["val_loss"], label="Validação (loss sem pesos)")
    axes[0].set(title="Loss por época", xlabel="Época (índice a partir de 0)", ylabel="Binary cross-entropy")
    axes[1].plot(history["pr_auc"], label="Treino")
    axes[1].plot(history["val_pr_auc"], label="Validação")
    axes[1].set(title="PR-AUC aproximada do Keras", xlabel="Época (índice a partir de 0)", ylabel="AUC(curve='PR')")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mlp_training_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, prediction, labels=[0, 1], display_labels=["No prazo", "Atrasado"],
        cmap="Blues", values_format=",d", ax=ax,
    )
    ax.set(title="MLP — matriz de confusão no teste", xlabel="Previsto", ylabel="Real")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mlp_confusion_matrix.png", dpi=160)
    plt.close(fig)
