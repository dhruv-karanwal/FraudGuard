"""Alert queue: every transfer the model blocked or sent to review."""
import streamlit as st

from app import core

core.header("Alert queue", "Every transfer the model blocked or sent for review. Filter the queue and select a row to open its case.")
df = core.transactions()
if df is None:
    st.info("No scored transactions yet.")
    st.stop()

alerts = core.with_status(df[df["tier"] != "APPROVE"])
lo, hi = int(df["step"].min()), int(df["step"].max())

filters = core.card("filters")
f = filters.columns([1, 1.2, 2.6])
tiers = f[0].pills("Decision", ["BLOCK", "REVIEW"], selection_mode="multi", default=["BLOCK", "REVIEW"])
types = f[1].pills("Type", ["TRANSFER", "CASH_OUT"], selection_mode="multi", default=["TRANSFER", "CASH_OUT"])
statuses = f[2].pills("Status", list(core.STATUSES), selection_mode="multi",
                      default=["New", "Investigating", "Escalated"])
g = filters.columns([1, 3])
min_amt = g[0].number_input("Min amount", min_value=0.0, value=0.0, step=10000.0)
hours = g[1].slider("Hours", lo, hi, (lo, hi))

view = alerts[alerts["tier"].isin(tiers) & alerts["type"].isin(types) & alerts["status"].isin(statuses)
              & (alerts["amount"] >= min_amt) & alerts["step"].between(*hours)]
view = view.sort_values(["score", "step"], ascending=False)

c = st.columns(4)
c[0].metric("In queue", f"{len(view):,}", border=True)
c[1].metric("Blocked", f"{int((view['tier'] == 'BLOCK').sum()):,}", border=True)
c[2].metric("For review", f"{int((view['tier'] == 'REVIEW').sum()):,}", border=True)
c[3].metric("Amount at risk", core.fmt_money(view["amount"].sum()), border=True)

if view.empty:
    st.success("Queue clear: no alerts match these filters.")
    st.stop()

st.caption("Select a row to open the case. Decision is shown as text as well as colour.")
event = st.dataframe(
    view[["row_id", "tier", "score", "step", "type", "amount", "nameOrig", "nameDest", "status"]],
    hide_index=True, width="stretch", height=460, on_select="rerun", selection_mode="single-row",
    column_config={
        "row_id": st.column_config.NumberColumn("Txn ID", format="%d"),
        "tier": "Decision",
        "score": core.risk_column(),
        "step": "Hour", "type": "Type", "nameOrig": "From", "nameDest": "To", "status": "Status",
        "amount": st.column_config.NumberColumn("Amount", format="%,.2f"),
    })

rows = event.selection.rows
if rows:
    sel = view.iloc[rows[0]]
    st.session_state["case_id"] = int(sel["row_id"])
    a, b = st.columns([3, 1])
    a.markdown(f"Selected {core.tier_chip(sel['tier'])} transfer **#{int(sel['row_id'])}** · "
               f"{core.fmt_money(sel['amount'])} from `{sel['nameOrig']}` to `{sel['nameDest']}`",
               unsafe_allow_html=True)
    if b.button("Open case", type="primary", icon=":material/policy:", width="stretch"):
        st.switch_page("views/investigate.py", query_params={"txn": int(sel["row_id"])})
