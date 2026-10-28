"""MLP regularizada, baseada na rede final do notebook de referência."""

from tensorflow import keras
from tensorflow.keras import layers

from module_deep_learning.config import LEARNING_RATE, SEED


def build_model(n_features):
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([layers.Input(shape=(n_features,))], name="mlp_olist")
    for units in (128, 64):
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    # Instâncias novas de métricas para cada modelo.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.AUC(curve="ROC", name="roc_auc")],
    )
    return model
