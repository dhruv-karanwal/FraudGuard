import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from src.models import build_dnn, build_hybrid, explainer  # noqa: E402
from src.training import set_seed  # noqa: E402

N_TAB, L, N_SEQ = 6, 4, 3


def _batch(b=8):
    rng = np.random.default_rng(42)
    tab = rng.normal(size=(b, N_TAB)).astype("float32")
    seq = rng.normal(size=(b, L, N_SEQ)).astype("float32")
    mask = np.zeros((b, L), dtype="float32")
    mask[:, -1] = 1
    mask[: b // 2, -3:] = 1
    seq[mask == 0] = 0
    return {"tab": tab, "seq": seq, "mask": mask}


@pytest.mark.parametrize("rnn", ["gru", "lstm"])
def test_hybrid_outputs_probabilities(rnn):
    set_seed()
    m = build_hybrid(N_TAB, L, N_SEQ, width=8, rnn=rnn)
    p = m.predict(_batch(), verbose=0)
    assert p.shape == (8, 1)
    assert ((p >= 0) & (p <= 1)).all()


def test_padding_values_do_not_change_prediction():
    set_seed()
    m = build_hybrid(N_TAB, L, N_SEQ, width=8)
    x = _batch()
    p1 = m.predict(x, verbose=0)
    x2 = {k: v.copy() for k, v in x.items()}
    x2["seq"][x2["mask"] == 0] = 123.0  # garbage in padded slots
    p2 = m.predict(x2, verbose=0)
    np.testing.assert_allclose(p1, p2, atol=1e-5)


def test_explainer_attention_weights_sum_to_one_and_skip_padding():
    set_seed()
    m = build_hybrid(N_TAB, L, N_SEQ, width=8)
    x = _batch()
    _, t_attn, f_attn = explainer(m).predict(x, verbose=0)
    np.testing.assert_allclose(t_attn.sum(1), 1, atol=1e-5)
    np.testing.assert_allclose(f_attn.sum(1), 1, atol=1e-5)
    assert (t_attn[x["mask"] == 0] < 1e-6).all()


def test_save_load_roundtrip(tmp_path):
    set_seed()
    m = build_hybrid(N_TAB, L, N_SEQ, width=8)
    x = _batch()
    path = tmp_path / "h.keras"
    m.save(path)
    m2 = keras.models.load_model(path)
    np.testing.assert_allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-6)


def test_dnn_only_shape():
    m = build_dnn(N_TAB, width=8)
    assert m.predict(_batch()["tab"], verbose=0).shape == (8, 1)
