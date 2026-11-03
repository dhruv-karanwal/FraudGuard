"""FraudGuard Streamlit dashboard.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.inference import Predictor, available_tag

st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="wide")

LABELS = {"logreg": "LogReg", "random_forest": "RandomForest", "xgboost": "XGBoost",
          "dnn_only": "DNN-only", "hybrid_gru": "Hybrid (DNN+GRU)", "hybrid_lstm": "Hybrid (DNN+LSTM)"}


# ---------- cached loaders ----------
@st.cache_resource(show_spinner="Loading model...")
def get_predictor(tag: str) -> Predictor:
    return Predictor(tag)


@st.cache_data
def read_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner="Loading test transactions...")
def test_table(tag: str) -> pd.DataFrame | None:
    d = config.PROCESSED_DIR / tag
    meta_p = d / "test_meta.parquet"
    scores_p = config.MODELS_DIR / tag / "hybrid_gru_test_scores.npy"
    if not meta_p.exists():
        return None
    df = pd.read_parquet(meta_p)
    if scores_p.exists():
        df["score"] = np.load(scores_p)
    return df


def figure(name: str, caption: str | None = None):
    p = config.FIGURES_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not generated yet. Run `python -m src.evaluate`.")


TAG = available_tag()


# ---------- pages ----------
def page_dashboard():
    st.title("🛡️ FraudGuard dashboard")
    st.caption("Hybrid DNN + GRU fraud detector on the PaySim mobile-money simulation.")
    eda = read_json(str(config.REPORTS_DIR / "eda_summary.json"))
    if eda:
        c = st.columns(4)
        c[0].metric("Transactions", f"{eda['rows'] / 1e6:.2f}M", help=f"{eda['rows']:,} rows")
        c[1].metric("Fraud cases", f"{eda['fraud_rows']:,}")
        c[2].metric("Fraud rate", f"{eda['fraud_rate']:.3%}")
        c[3].metric("Simulated hours", f"{eda['steps'][1]}")
        b = eda["split_boundaries"]
        st.markdown(
            f"**Chronological split** (TRANSFER + CASH_OUT rows): train steps 1-{b['last_train_step']}, "
            f"validation {b['last_train_step'] + 1}-{b['last_val_step']}, test {b['last_val_step'] + 1}-{eda['steps'][1]}.")

    df = test_table(TAG) if TAG else None
    if df is not None and "score" in df:
        thr = get_predictor(TAG).threshold
        df["flagged"] = df["score"] >= thr
        st.subheader("Test period: model alerts")
        c = st.columns(4)
        c[0].metric("Scored test txns", f"{len(df) / 1e3:,.0f}k", help=f"{len(df):,} TRANSFER/CASH_OUT rows")
        c[1].metric("Alerts raised", f"{int(df.flagged.sum()):,}")
        caught, total = int((df.flagged & (df.isFraud == 1)).sum()), int(df.isFraud.sum())
        c[2].metric("Fraud caught", f"{caught / total:.1%}", help=f"{caught:,} of {total:,} fraud transactions")
        c[3].metric("False alarms", f"{int((df.flagged & (df.isFraud == 0)).sum()):,}")
        per_step = df.groupby("step").agg(alerts=("flagged", "sum"), fraud=("isFraud", "sum"))
        st.caption("Alerts per hour closely track actual fraud per hour.")
        st.line_chart(per_step, height=260, color=["#F58518", "#4C78A8"])
        st.markdown("**Highest-risk test transactions**")
        top = df.sort_values("score", ascending=False).head(25)
        st.dataframe(top[["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                          "nameDest", "score", "isFraud"]].style.format({"amount": "{:,.2f}",
                          "oldbalanceOrg": "{:,.2f}", "newbalanceOrig": "{:,.2f}", "score": "{:.4f}"}),
                     width="stretch", hide_index=True)
    else:
        st.info("Test scores not available yet. Run `python -m src.evaluate`.")

    st.subheader("Data")
    c1, c2 = st.columns(2)
    with c1:
        figure("class_distribution.png")
    with c2:
        figure("amount_distribution.png")
    figure("fraud_over_time.png")


def page_predict():
    st.title("🔎 Predict a transaction")
    if not TAG:
        st.error("No trained hybrid model found. Run `python -m src.train`.")
        return
    pred = get_predictor(TAG)
    st.caption(f"Model: hybrid DNN + GRU (`models/{TAG}`); decision threshold {pred.threshold:.4f} "
               "(tuned on validation). Only TRANSFER and CASH_OUT are scored: fraud never occurs in other types.")

    df = test_table(TAG)
    if df is not None:
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("Load a test fraud example"):
            st.session_state["example"] = df[df.isFraud == 1].sample(1, random_state=None).iloc[0].to_dict()
        if c2.button("Load a test legit example"):
            st.session_state["example"] = df[df.isFraud == 0].sample(1, random_state=None).iloc[0].to_dict()
        # Demo link: ?page=predict&example=fraud|legit loads a fixed example and scores it.
        qp = str(st.query_params.get("example", "")).lower()
        if qp in ("fraud", "legit") and not st.session_state.get("demo_done"):
            pool = df[df.isFraud == (1 if qp == "fraud" else 0)]
            st.session_state["example"] = pool.sample(1, random_state=config.SEED).iloc[0].to_dict()
            st.session_state["demo_done"] = True
            st.session_state["autoscore"] = True
    ex = st.session_state.get("example", {})

    with st.form("txn"):
        c = st.columns(3)
        types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
        ttype = c[0].selectbox("Type", types, index=types.index(ex.get("type", "TRANSFER")))
        step = c[1].number_input("Step (hour of simulation)", 1, 1000, int(ex.get("step", 400)))
        amount = c[2].number_input("Amount", 0.0, value=float(ex.get("amount", 181.0)), format="%.2f")
        c = st.columns(4)
        ob_o = c[0].number_input("Origin balance before", 0.0, value=float(ex.get("oldbalanceOrg", 181.0)), format="%.2f")
        nb_o = c[1].number_input("Origin balance after", 0.0, value=float(ex.get("newbalanceOrig", 0.0)), format="%.2f")
        ob_d = c[2].number_input("Destination balance before", 0.0, value=float(ex.get("oldbalanceDest", 0.0)), format="%.2f")
        nb_d = c[3].number_input("Destination balance after", 0.0, value=float(ex.get("newbalanceDest", 0.0)), format="%.2f")

        st.markdown(f"**Earlier transactions of this account** (optional, up to {pred.seq_len - 1}, oldest first)")
        hist_default = pd.DataFrame(columns=["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                                             "oldbalanceDest", "newbalanceDest"])
        hist = st.data_editor(hist_default, num_rows="dynamic", width="stretch",
                              column_config={"type": st.column_config.SelectboxColumn(options=types)})
        submitted = st.form_submit_button("Score transaction", type="primary")

    if not (submitted or st.session_state.pop("autoscore", False)):
        return
    current = {"step": step, "type": ttype, "amount": amount, "oldbalanceOrg": ob_o, "newbalanceOrig": nb_o,
               "oldbalanceDest": ob_d, "newbalanceDest": nb_d}
    history = [r for r in hist.dropna(how="any").to_dict("records")]
    bad = [r for r in history if r["step"] > step]
    if bad:
        st.error("History rows must have a step at or before the current transaction.")
        return
    history.sort(key=lambda r: r["step"])
    res = pred.predict(current, history)

    if not res.scored:
        st.success(f"{ttype} transactions are not scored: no fraud of this type exists in PaySim.")
        return
    c = st.columns(3)
    c[0].metric("Fraud probability", f"{res.probability:.2%}")
    c[1].metric("Decision", "🚨 FRAUD" if res.is_fraud else "✅ Legit")
    c[2].metric("History items used", f"{res.window_len - 1}")
    st.progress(min(1.0, res.probability))
    if ex and ex.get("type") == ttype and float(ex.get("amount", -1)) == amount:
        st.caption(f"Loaded example's true label: {'fraud' if ex.get('isFraud') == 1 else 'legit'}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fusion attention** (how much each branch drove the score)")
        st.bar_chart(pd.Series(res.fusion_weights, name="weight"), height=220)
    with c2:
        st.markdown("**Temporal attention** over the account window (oldest → current)")
        w = res.time_attention[-res.window_len:]
        labels = [f"t-{res.window_len - 1 - i}" if i < res.window_len - 1 else "current" for i in range(res.window_len)]
        st.bar_chart(pd.Series(w, index=labels, name="weight"), height=220)

    err_o = nb_o + amount - ob_o
    err_d = ob_d + amount - nb_d
    st.markdown("**Signals in this transaction**")
    st.write(pd.DataFrame({
        "signal": ["Origin account fully drained (amount = balance before)", "Origin balance error",
                   "Destination balance error", "Destination balances both zero"],
        "value": [str(abs(amount - ob_o) < 0.01 and ob_o > 0), f"{err_o:,.2f}", f"{err_d:,.2f}",
                  str(ob_d == 0 and nb_d == 0)],
    }))


def page_performance():
    st.title("📊 Model performance")
    if not TAG:
        st.error("No trained models found.")
        return
    res = read_json(str(config.REPORTS_DIR / f"{TAG}_test_metrics.json"))
    if not res:
        st.info("Test metrics not available yet. Run `python -m src.evaluate`.")
        return
    st.caption(f"Test period only; each threshold was chosen on validation (data: `{TAG}`).")
    rows = []
    for n, m in res.items():
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        rows.append({"model": LABELS.get(n, n), "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "threshold": m["threshold"],
                     "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    table = pd.DataFrame(rows).set_index("model")
    st.dataframe(table.style.format({c: "{:.4f}" for c in ["precision", "recall", "f1", "roc_auc", "pr_auc",
                                                            "threshold"]})
                 .highlight_max(subset=["precision", "recall", "f1", "roc_auc", "pr_auc"], color="#d4edda"),
                 width="stretch")

    tabs = st.tabs(["Comparison", "Confusion matrices", "ROC / PR", "Training", "Architecture"])
    with tabs[0]:
        figure("model_comparison.png")
    with tabs[1]:
        figure("confusion_matrix_hybrid.png", "Hybrid model; layout [[TN, FP], [FN, TP]]")
        cols = st.columns(2)
        for i, n in enumerate([n for n in res if not n.startswith("hybrid_gru")]):
            with cols[i % 2]:
                figure(f"confusion_matrix_{n}.png")
    with tabs[2]:
        figure("roc_pr_curves.png")
    with tabs[3]:
        figure("training_curves.png")
    with tabs[4]:
        figure("architecture.png")
        summ = config.REPORTS_DIR / "hybrid_model_summary.txt"
        if summ.exists():
            st.code(summ.read_text(encoding="utf-8"), language="text")


PAGES = {"Dashboard": page_dashboard, "Predict Transaction": page_predict, "Model Performance": page_performance}
SLUGS = {"dashboard": 0, "predict": 1, "performance": 2}  # deep links: ?page=predict
with st.sidebar:
    st.header("FraudGuard")
    start = SLUGS.get(str(st.query_params.get("page", "")).lower(), 0)
    choice = st.radio("Page", list(PAGES), index=start)
    st.caption(f"Artifacts: `{TAG or 'none'}`")
PAGES[choice]()
