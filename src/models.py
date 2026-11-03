"""Keras models: DNN-only baseline and the hybrid DNN + GRU/LSTM attention model."""
from __future__ import annotations

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable("fraudguard")
class MaskedAttentionPool(layers.Layer):
    """Additive attention over time steps, ignoring padded positions.

    Returns (context, weights) with shapes (b, d) and (b, L).
    """

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def build(self, h_shape, mask_shape=None):
        self.proj.build(h_shape)
        self.score.build((*h_shape[:-1], self.units))

    def compute_mask(self, *args, **kwargs):
        return None  # mask is consumed here

    def call(self, h, mask):
        e = ops.squeeze(self.score(self.proj(h)), axis=-1)
        e = e + (1.0 - mask) * -1e9
        a = ops.softmax(e, axis=-1)
        ctx = ops.sum(h * ops.expand_dims(a, -1), axis=1)
        return ctx, a

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable("fraudguard")
class FusionAttention(layers.Layer):
    """Attention over branch embeddings (b, k, d) -> fused (b, d), weights (b, k)."""

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def build(self, shape):
        self.proj.build(shape)
        self.score.build((*shape[:-1], self.units))

    def call(self, branches):
        e = ops.squeeze(self.score(self.proj(branches)), axis=-1)
        w = ops.softmax(e, axis=-1)
        fused = ops.sum(branches * ops.expand_dims(w, -1), axis=1)
        return fused, w

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable("fraudguard")
class StackBranches(layers.Layer):
    def call(self, tensors):
        return ops.stack(tensors, axis=1)


def _tab_branch(x, width: int, dropout: float):
    x = layers.Dense(128, name="tab_dense1")(x)
    x = layers.BatchNormalization(name="tab_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", name="tab_dense2")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(width, activation="relu", name="tab_embed")(x)


def _metrics():
    return [keras.metrics.AUC(curve="PR", name="pr_auc"), keras.metrics.AUC(name="roc_auc"),
            keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]


def build_dnn(n_tab: int, width: int = 32, dropout: float = 0.2, lr: float = 1e-3) -> keras.Model:
    tab_in = keras.Input((n_tab,), name="tab")
    h = _tab_branch(tab_in, width, dropout)
    out = layers.Dense(1, activation="sigmoid", name="fraud")(h)
    model = keras.Model(tab_in, out, name="dnn_only")
    model.compile(keras.optimizers.Adam(lr), "binary_crossentropy", metrics=_metrics())
    return model


def build_hybrid(n_tab: int, seq_len: int, n_seq: int, width: int = 32, rnn: str = "gru",
                 dropout: float = 0.2, lr: float = 1e-3) -> keras.Model:
    """DNN branch (current transaction) + GRU/LSTM branch (account history),
    each pooled to ``width`` dims, fused by attention, sigmoid output."""
    tab_in = keras.Input((n_tab,), name="tab")
    seq_in = keras.Input((seq_len, n_seq), name="seq")
    mask_in = keras.Input((seq_len,), name="mask")

    h_tab = _tab_branch(tab_in, width, dropout)

    rnn_layer = {"gru": layers.GRU, "lstm": layers.LSTM}[rnn]
    h_seq = rnn_layer(width, return_sequences=True, name=f"seq_{rnn}")(
        seq_in, mask=ops.cast(mask_in, "bool"))
    h_seq, _ = MaskedAttentionPool(width, name="time_attention")(h_seq, mask_in)
    h_seq = layers.Dense(width, activation="relu", name="seq_embed")(h_seq)

    branches = StackBranches(name="stack_branches")([h_tab, h_seq])
    fused, _ = FusionAttention(width, name="fusion_attention")(branches)
    x = layers.Dense(32, activation="relu", name="head_dense")(fused)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="fraud")(x)

    model = keras.Model([tab_in, seq_in, mask_in], out, name=f"hybrid_{rnn}")
    model.compile(keras.optimizers.Adam(lr), "binary_crossentropy", metrics=_metrics())
    return model


def explainer(model: keras.Model) -> keras.Model:
    """Model returning (probability, time-attention weights, fusion weights)."""
    t_attn = model.get_layer("time_attention").output[1]
    f_attn = model.get_layer("fusion_attention").output[1]
    return keras.Model(model.inputs, [model.output, t_attn, f_attn])
