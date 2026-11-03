"""Shared training helpers for the Keras models."""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np

from src import config


def set_seed(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def fit_keras(model, x_train, y_train, x_val, y_val, out_path, class_weight,
              epochs: int = 30, batch_size: int = 2048, patience: int = 5) -> dict:
    """Fit with class weights and early stopping on validation PR-AUC.

    Saves the best model to ``out_path`` and the history next to it.
    """
    import keras

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5,
                                          patience=2, min_lr=1e-5),
    ]
    t0 = time.time()
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                     batch_size=batch_size, class_weight=class_weight, callbacks=cbs,
                     shuffle=True, verbose=2)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    history["train_seconds"] = time.time() - t0
    model.save(out_path)
    with open(str(out_path).replace(".keras", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
