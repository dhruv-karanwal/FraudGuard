"""Headless smoke test of every Streamlit page. Needs trained artifacts."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pytest

from src.inference import available_tag

pytestmark = pytest.mark.skipif(available_tag() is None, reason="no trained hybrid model")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _open(page: str):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", ["Dashboard", "Predict Transaction", "Model Performance"])
def test_page_renders_without_exception(page):
    at = _open(page)
    assert len(at.title) == 1


def test_predict_scores_a_transaction():
    at = _open("Predict Transaction")
    # Default form values describe a drained-account TRANSFER, the classic fraud pattern.
    submit = [b for b in at.button if b.label == "Score transaction"][0]
    submit.click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Fraud probability" in labels and "Decision" in labels


def test_performance_lists_all_models():
    at = _open("Model Performance")
    df = at.dataframe[0].value
    assert {"LogReg", "RandomForest", "XGBoost", "DNN-only", "Hybrid (DNN+GRU)"} <= set(df.index)
