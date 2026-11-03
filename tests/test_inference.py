"""App inference must reproduce the batch pipeline exactly. Needs dev artifacts
(python -m src.prepare --dev && python -m src.train --dev); skipped otherwise."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pytest

from src import config

DEV_MODEL = config.MODELS_DIR / "dev" / "hybrid_gru.keras"
pytestmark = pytest.mark.skipif(not DEV_MODEL.exists(), reason="dev artifacts not built")


@pytest.fixture(scope="module")
def predictor():
    from src.inference import Predictor
    return Predictor("dev")


@pytest.fixture(scope="module")
def dev_test():
    from src.prepare import load_split
    d = load_split("dev", "test")
    meta = pd.read_parquet(config.PROCESSED_DIR / "dev" / "test_meta.parquet")
    return d, meta


def test_single_row_matches_batch_features(predictor, dev_test):
    d, meta = dev_test
    rows = np.flatnonzero(d["mask"].sum(1) == 1)[:20]
    for i in rows:
        cur = meta.iloc[i].to_dict()
        x = predictor.arrays(cur)
        np.testing.assert_allclose(x["tab"][0], d["tab"][i], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(x["seq"][0], d["seq"][i], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(x["mask"][0], d["mask"][i])


def test_single_row_matches_batch_scores(predictor, dev_test):
    d, meta = dev_test
    rows = np.flatnonzero(d["mask"].sum(1) == 1)[:50]
    batch = predictor.predict_batch({k: d[k][rows] for k in ("tab", "seq", "mask")})
    single = [predictor.predict(meta.iloc[i].to_dict()).probability for i in rows]
    np.testing.assert_allclose(single, batch, rtol=1e-4, atol=1e-6)


def test_unscored_type_returns_zero(predictor):
    p = predictor.predict({"step": 1, "type": "PAYMENT", "amount": 10.0, "oldbalanceOrg": 20.0,
                           "newbalanceOrig": 10.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0})
    assert not p.scored and p.probability == 0.0


def test_history_is_used_and_attention_is_valid(predictor):
    cur = {"step": 10, "type": "TRANSFER", "amount": 500.0, "oldbalanceOrg": 500.0, "newbalanceOrig": 0.0,
           "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
    hist = [{**cur, "step": 5, "type": "PAYMENT", "amount": 20.0, "oldbalanceOrg": 520.0, "newbalanceOrig": 500.0}]
    p = predictor.predict(cur, hist)
    assert p.window_len == 2
    assert sum(p.time_attention) == pytest.approx(1, abs=1e-4)
    assert sum(p.fusion_weights.values()) == pytest.approx(1, abs=1e-4)
    assert all(w < 1e-6 for w in p.time_attention[:-2])
