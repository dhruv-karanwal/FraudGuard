"""Check transaction: score one transfer like a payment-gateway call."""
import time

import pandas as pd
import streamlit as st

from app import core
from src import config

st.title("Check transaction")
pred = core.predictor()
st.caption("Score a transfer before it settles. Only TRANSFER and CASH_OUT are scored, because fraud never "
           "occurs in other types. Decision tiers: BLOCK at or above the validation-tuned threshold, "
           f"REVIEW from {core.REVIEW_FLOOR:.2f}, otherwise APPROVE.")

df = core.transactions()
if df is not None:
    c1, c2, _ = st.columns([1.4, 1.4, 2])
    if c1.button("Load a test fraud example", icon=":material/gpp_bad:"):
        st.session_state["example"] = df[df.isFraud == 1].sample(1).iloc[0].to_dict()
    if c2.button("Load a test legit example", icon=":material/verified_user:"):
        st.session_state["example"] = df[df.isFraud == 0].sample(1).iloc[0].to_dict()
    # Demo link: ?example=fraud|legit loads a fixed example (seed 42) and scores it.
    qp = str(st.query_params.get("example", "")).lower()
    if qp in ("fraud", "legit") and not st.session_state.get("demo_done"):
        pool = df[df.isFraud == (1 if qp == "fraud" else 0)]
        st.session_state["example"] = pool.sample(1, random_state=config.SEED).iloc[0].to_dict()
        st.session_state["demo_done"] = True
        st.session_state["autoscore"] = True
ex = st.session_state.get("example", {})

types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
with st.form("txn"):
    c = st.columns(3)
    ttype = c[0].selectbox("Type", types, index=types.index(ex.get("type", "TRANSFER")))
    step = c[1].number_input("Hour of simulation", 1, 1000, int(ex.get("step", 400)))
    amount = c[2].number_input("Amount", 0.0, value=float(ex.get("amount", 181.0)), format="%.2f")
    c = st.columns(4)
    ob_o = c[0].number_input("Origin balance before", 0.0, value=float(ex.get("oldbalanceOrg", 181.0)), format="%.2f")
    nb_o = c[1].number_input("Origin balance after", 0.0, value=float(ex.get("newbalanceOrig", 0.0)), format="%.2f")
    ob_d = c[2].number_input("Destination balance before", 0.0, value=float(ex.get("oldbalanceDest", 0.0)),
                             format="%.2f")
    nb_d = c[3].number_input("Destination balance after", 0.0, value=float(ex.get("newbalanceDest", 0.0)),
                             format="%.2f")
    st.markdown(f"**Earlier transfers of this account** (optional, up to {pred.seq_len - 1}, oldest first)")
    hist = st.data_editor(
        pd.DataFrame(columns=["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                              "oldbalanceDest", "newbalanceDest"]),
        num_rows="dynamic", width="stretch", column_config={"type": st.column_config.SelectboxColumn(options=types)})
    submitted = st.form_submit_button("Score transaction", type="primary", icon=":material/bolt:")

if not (submitted or st.session_state.pop("autoscore", False)):
    st.stop()

current = {"step": step, "type": ttype, "amount": amount, "oldbalanceOrg": ob_o, "newbalanceOrig": nb_o,
           "oldbalanceDest": ob_d, "newbalanceDest": nb_d}
history = sorted(hist.dropna(how="any").to_dict("records"), key=lambda r: r["step"])
if any(r["step"] > step for r in history):
    st.error("Earlier transfers must have an hour at or before the current transaction.")
    st.stop()

t0 = time.perf_counter()
res = pred.predict(current, history)
latency = (time.perf_counter() - t0) * 1000

if not res.scored:
    core.decision_card("APPROVE", 0.0, latency)
    st.caption(f"{ttype} transactions are approved without scoring: PaySim has no fraud of this type.")
    st.stop()

decision = str(core.tier([res.probability], res.threshold)[0])
core.decision_card(decision, res.probability, latency)
c = st.columns(3)
c[0].metric("Fraud probability", f"{res.probability:.2%}", border=True)
c[1].metric("Decision", decision, border=True)
c[2].metric("History items used", f"{res.window_len - 1}", border=True)
if ex and ex.get("type") == ttype and float(ex.get("amount", -1)) == amount:
    st.caption(f"Loaded example's true label: {'fraud' if ex.get('isFraud') == 1 else 'legit'}")

left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**Why**")
    large = float(df["amount"].quantile(0.99)) if df is not None else float("inf")
    sig = core.signals(current, large)
    st.markdown(" ".join(core.chip(s, "#7F1D1D") for s in sig) or "No rule-based risk signals.",
                unsafe_allow_html=True)
    fw = res.fusion_weights
    st.progress(fw["transaction"], text=f"Driven by this transaction: {fw['transaction']:.0%}")
    st.progress(fw["history"], text=f"Driven by account history: {fw['history']:.0%}")
with right, st.container(border=True):
    st.markdown("**Temporal attention** over the account window (oldest → current)")
    w = res.time_attention[-res.window_len:]
    labels = [f"t-{res.window_len - 1 - i}" if i < res.window_len - 1 else "current" for i in range(res.window_len)]
    st.bar_chart(pd.Series(w, index=labels, name="weight"), height=180, color="#38BDF8")

with st.expander("API response"):
    st.json({"decision": decision, "fraud_probability": round(res.probability, 6),
             "block_threshold": round(res.threshold, 6), "review_floor": core.REVIEW_FLOOR,
             "signals": sig, "attention": {k: round(v, 4) for k, v in fw.items()},
             "latency_ms": round(latency, 1), "model": f"{core.MODEL}/{core.TAG}"})
