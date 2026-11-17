"""Headless smoke tests of every console page, plus the business rules. Page tests need artifacts."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pytest

from src.inference import available_tag

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
needs_model = pytest.mark.skipif(available_tag() is None, reason="no trained hybrid model")
PAGES = ["overview", "alerts", "investigate", "live", "check", "model"]


def _open(page: str):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if page != "overview":
        at.switch_page(f"views/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@needs_model
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = _open(page)
    assert len(at.title) == 1


@needs_model
def test_check_scores_a_transaction():
    at = _open("check")
    # Default form values describe a drained-account TRANSFER, the classic fraud pattern.
    [b for b in at.button if b.label == "Score transaction"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Fraud probability" in labels and "Decision" in labels


@needs_model
def test_investigate_records_analyst_decision():
    at = _open("investigate")
    [b for b in at.button if b.label == "Confirm fraud"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Confirmed fraud" in [c["status"] for c in at.session_state["cases"].values()]


@needs_model
def test_live_monitor_advances_when_started():
    at = _open("live")
    [b for b in at.button if b.label == "Start"][0].click().run()
    assert at.session_state["live_running"]


@needs_model
def test_model_page_lists_all_models():
    at = _open("model")
    labels = {i for df in at.dataframe for i in df.value.index}
    assert {"LogReg", "RandomForest", "XGBoost", "DNN-only", "Hybrid (DNN+GRU)"} <= labels


# ---------- business rules (no artifacts needed) ----------
from app import core  # noqa: E402


def test_tiers_follow_thresholds():
    assert core.tier([0.99, 0.6, 0.1], 0.9).tolist() == ["BLOCK", "REVIEW", "APPROVE"]


def test_money_counts_and_amounts():
    df = pd.DataFrame({"score": [0.95, 0.95, 0.2, 0.1], "isFraud": [1, 0, 1, 0], "amount": [100.0, 5.0, 40.0, 1.0]})
    m = core.money(df, 0.9)
    assert (m["caught"], m["false_alarms"], m["missed"], m["alerts"]) == (1, 1, 1, 2)
    assert (m["protected"], m["lost"]) == (100.0, 40.0)


def test_signals_flag_drained_account():
    row = {"type": "TRANSFER", "amount": 500.0, "oldbalanceOrg": 500.0, "newbalanceOrig": 0.0,
           "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
    s = core.signals(row)
    assert "Origin account fully drained" in s and "Destination balance unchanged" in s


def _random_scores(n=300):
    rng = np.random.default_rng(42)
    return pd.DataFrame({"score": rng.random(n).round(2), "isFraud": rng.integers(0, 2, n),
                         "amount": rng.random(n) * 100})


def test_cost_curve_matches_money_at_every_threshold():
    df = _random_scores()
    curve = core.cost_curve(df, false_alarm_cost=7.0, min_threshold=0.0)
    for _, r in curve.sample(20, random_state=42).iterrows():
        m = core.money(df, r["threshold"])
        assert r["false_alarms"] == m["false_alarms"] and r["missed"] == m["missed"]
        assert r["total_cost"] == pytest.approx(m["lost"] + 7.0 * m["false_alarms"], abs=0.01)  # cents


def test_cost_curve_optimum_beats_any_threshold():
    df = _random_scores()
    curve = core.cost_curve(df, false_alarm_cost=20.0, min_threshold=0.0)
    best = curve["total_cost"].min()
    for t in (0.1, 0.33, 0.5, 0.77, 0.95):
        m = core.money(df, t)
        assert best <= m["lost"] + 20.0 * m["false_alarms"] + 1e-9
