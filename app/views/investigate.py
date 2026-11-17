"""Case investigation: one alert, why it was flagged, and the analyst's decision."""
import pandas as pd
import streamlit as st

from app import core

st.title("Investigate")
df = core.transactions()
if df is None:
    st.info("No scored transactions yet.")
    st.stop()

alerts = df[df["tier"] != "APPROVE"].sort_values("score", ascending=False)
qp = st.query_params.get("txn")
wanted = int(qp) if qp and str(qp).isdigit() else st.session_state.get("case_id")
ids = alerts["row_id"].astype(int).tolist()
if wanted not in ids:
    wanted = ids[0]
choice = st.selectbox("Case", ids, index=ids.index(wanted),
                      format_func=lambda r: f"#{r} · {core.case(r)['status']}")
st.session_state["case_id"] = choice
st.query_params["txn"] = choice
row = df[df["row_id"] == choice].iloc[0].to_dict()
large = float(df["amount"].quantile(0.99))

# History of this origin account in the monitoring window (rarely more than one row in PaySim).
history = df[(df["nameOrig"] == row["nameOrig"]) & (df["row_id"] < choice)].sort_values("row_id")
res = core.predictor().predict(row, history.to_dict("records"))

top = st.columns([1.5, 1])
with top[0]:
    core.decision_card(row["tier"], res.probability)
    st.markdown(f"**{row['type']}** of **{row['amount']:,.2f}** at hour **{int(row['step'])}**  \n"
                f"From `{row['nameOrig']}` → To `{row['nameDest']}`")
    bal = pd.DataFrame({
        "Account": ["Origin", "Destination"],
        "Balance before": [row["oldbalanceOrg"], row["oldbalanceDest"]],
        "Balance after": [row["newbalanceOrig"], row["newbalanceDest"]],
    })
    st.dataframe(bal, hide_index=True, width="stretch", column_config={
        "Balance before": st.column_config.NumberColumn(format="%,.2f"),
        "Balance after": st.column_config.NumberColumn(format="%,.2f")})

with top[1]:
    with st.container(border=True):
        st.markdown("**Why it was flagged**")
        sig = core.signals(row, large)
        st.markdown(" ".join(core.chip(s, "#7F1D1D") for s in sig) or "No rule-based signals.",
                    unsafe_allow_html=True)
        st.markdown("**What drove the model**")
        fw = res.fusion_weights
        st.progress(fw["transaction"], text=f"This transaction: {fw['transaction']:.0%}")
        st.progress(fw["history"], text=f"Account history: {fw['history']:.0%} "
                                        f"({res.window_len - 1} earlier transfer(s))")
        st.markdown('<span class="fg-muted">Signals are plain-language checks on the balances; '
                    'the percentages are the model\'s own attention weights.</span>', unsafe_allow_html=True)

st.subheader("Destination account activity")
dest = df[df["nameDest"] == row["nameDest"]].sort_values("step")
st.caption(f"`{row['nameDest']}` received {len(dest)} scored transfer(s) in the monitoring window. "
           "Many incoming transfers to one account can indicate a mule account.")
st.dataframe(dest[["row_id", "step", "type", "amount", "nameOrig", "tier", "score"]], hide_index=True,
             width="stretch", column_config={
                 "row_id": st.column_config.NumberColumn("Txn ID", format="%d"), "step": "Hour", "type": "Type",
                 "nameOrig": "From", "tier": "Decision",
                 "amount": st.column_config.NumberColumn("Amount", format="%,.2f"),
                 "score": core.risk_column()})

st.subheader("Analyst decision")
cur = core.case(choice)
with st.form("decision"):
    note = st.text_area("Notes", value=cur["note"], placeholder="What did you check? Who did you contact?")
    buttons = [("Confirm fraud", "Confirmed fraud", ":material/gpp_bad:"),
               ("Mark legit", "False positive", ":material/verified_user:"),
               ("Escalate", "Escalated", ":material/priority_high:"),
               ("Start investigating", "Investigating", ":material/search:")]
    status = None
    for col, (label, new_status, icon) in zip(st.columns(4), buttons):
        if col.form_submit_button(label, icon=icon, width="stretch"):
            status = new_status
if status:
    core.set_case(choice, status, note)
    st.toast(f"Case #{choice} marked: {status}", icon=":material/check_circle:")
st.markdown(f"Current status: {core.chip(core.case(choice)['status'])}"
            f'<span class="fg-muted">Ground truth (for this demo): '
            f'{"fraud" if row["isFraud"] == 1 else "legit"}</span>', unsafe_allow_html=True)
