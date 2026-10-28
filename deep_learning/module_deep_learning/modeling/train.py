"""Treinamento com pesos de classe e parada antecipada na validação."""

import json

from loguru import logger
import numpy as np
import tensorflow as tf
from tensorflow import keras

from module_deep_learning.config import BATCH_SIZE, EPOCHS, MODELS_DIR, PATIENCE, REPORTS_DIR, SEED
from module_deep_learning.modeling.model import build_model


def make_dataset(X, y=None, batch_size=BATCH_SIZE, training=False):
    tensors = np.asarray(X, dtype="float32")
    if y is not None:
        tensors = (tensors, np.asarray(y, dtype="float32"))
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        dataset = dataset.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    # Limita a fila de entrada; permite rodar em computadores com poucos recursos.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(batch_size).with_options(options).prefetch(1)


def train_model(X_train, y_train, X_val, y_val, epochs=EPOCHS):
    keras.utils.set_random_seed(SEED)
    keras.backend.clear_session()
    labels = np.asarray(y_train, dtype=int)
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("O treino precisa ter exemplos das duas classes.")
    class_weight = {i: float(len(labels) / (2 * counts[i])) for i in (0, 1)}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mlp_olist.keras"
    model = build_model(X_train.shape[1])
    logger.info(f"Arquitetura: entrada={X_train.shape[1]} → 128 → 64 → 1 | Pesos: {class_weight}")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_pr_auc", mode="max", save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
    ]
    result = model.fit(
        make_dataset(X_train, y_train, training=True),
        validation_data=make_dataset(X_val, y_val),
        class_weight=class_weight, epochs=epochs, callbacks=callbacks, verbose=2,
    )
    history = {name: [float(v) for v in values] for name, values in result.history.items()}
    (REPORTS_DIR / "mlp_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_epoch = int(np.argmax(history["val_pr_auc"]) + 1)
    logger.success(f"Treino concluído | Épocas: {len(history['loss'])} | Melhor época na validação: {best_epoch}")
    # Avalia exatamente o checkpoint salvo, com os pesos da melhor época.
    return keras.models.load_model(model_path), history, class_weight, best_epoch
