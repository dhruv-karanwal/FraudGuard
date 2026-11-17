"""Overview: business outcomes of the model over the monitoring window."""
import altair as alt
import pandas as pd
import streamlit as st

from app import core

st.title("Overview")
df = core.transactions()
if df is None:
    st.info("No scored transactions yet. Run `python -m src.evaluate`.")
    st.stop()

thr = core.threshold()
m = core.money(df, thr)
last_step = int(df["step"].max())
today = core.money(df[df["step"] > last_step - 24], thr)
prev = core.money(df[(df["step"] > last_step - 48) & (df["step"] <= last_step - 24)], thr)

st.caption(f"Monitoring window: hours {int(df['step'].min())}–{last_step}. "
           "“Last 24 h” compares the latest 24 simulated hours with the 24 before.")
c = st.columns(5)
c[0].metric("Money protected", core.fmt_money(m["protected"]), border=True,
            delta=f"{core.fmt_money(today['protected'] - prev['protected'])} last 24 h", delta_color="off",
            help="Total amount of fraudulent transfers the model blocked.")
c[1].metric("Fraud caught", f"{m['caught'] / m['fraud']:.1%}", border=True,
            help=f"{m['caught']:,} of {m['fraud']:,} fraudulent transfers blocked.")
c[2].metric("Alerts raised", f"{m['alerts']:,}", border=True,
            delta=f"{today['alerts'] - prev['alerts']:+,} last 24 h", delta_color="off")
c[3].metric("False alarms", f"{m['false_alarms']:,}", border=True,
            help=f"Legit transfers blocked: {m['false_alarms'] / max(m['alerts'], 1):.2%} of alerts.")
c[4].metric("Fraud missed", core.fmt_money(m["lost"]), border=True,
            help=f"{m['missed']:,} fraudulent transfers slipped through.")

left, right = st.columns([2.2, 1])
with left:
    st.subheader("Alerts vs actual fraud per hour")
    df["blocked"] = df["tier"] == "BLOCK"
    df["missed"] = (~df["blocked"]) & (df["isFraud"] == 1)
    hourly = df.groupby("step").agg(Alerts=("blocked", "sum"), Fraud=("isFraud", "sum"),
                                    Missed=("missed", "sum")).reset_index()
    long = hourly.melt("step", ["Alerts", "Fraud"], var_name="Series", value_name="Count")
    lines = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=alt.X("step:Q", title="Simulated hour"),
        y=alt.Y("Count:Q", title="Transfers per hour"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Alerts", "Fraud"], range=[core.INK, core.SUBTLE])),
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Alerts", "Fraud"], range=[[1, 0], [4, 3]])),
        tooltip=["step", "Series", "Count"])
    miss = alt.Chart(hourly[hourly["Missed"] > 0]).mark_point(shape="triangle-down", size=90, filled=True,
                                                             color=core.TIER_COLOR["BLOCK"]).encode(
        x="step:Q", y=alt.Y("Fraud:Q"), tooltip=[alt.Tooltip("step", title="Hour"),
                                                 alt.Tooltip("Missed", title="Missed fraud")])
    st.altair_chart((lines + miss).properties(height=300), width="stretch")
    n_miss_hours = int((hourly["Missed"] > 0).sum())
    st.caption(f"Red triangles mark the {n_miss_hours} hours with at least one missed fraud. "
               "Solid line = model alerts, dashed line = actual fraud.")

with right:
    st.subheader("Decisions")
    counts = df["tier"].value_counts().reindex(core.TIERS, fill_value=0).reset_index()
    counts.columns = ["Decision", "Transfers"]
    bar = alt.Chart(counts).mark_bar().encode(
        x=alt.X("Transfers:Q", scale=alt.Scale(type="symlog"), title="Transfers (log scale)"),
        y=alt.Y("Decision:N", sort=list(core.TIERS), title=None),
        color=alt.Color("Decision:N", legend=None,
                        scale=alt.Scale(domain=list(core.TIERS), range=[core.TIER_COLOR[t] for t in core.TIERS])),
        tooltip=["Decision", "Transfers"])
    text = bar.mark_text(align="left", dx=4, color=core.INK).encode(text=alt.Text("Transfers:Q", format=","))
    st.altair_chart((bar + text).properties(height=160), width="stretch")
    by_type = df[df["blocked"]].groupby("type").size().rename("Blocked")
    st.markdown("**Blocked by type**")
    for t, v in by_type.items():
        st.markdown(f"{core.chip(t)} {v:,}", unsafe_allow_html=True)
    open_cases = int((core.with_status(df[df["tier"] != "APPROVE"])["status"] == "New").sum())
    st.page_link("views/alerts.py", label=f"Open alert queue ({open_cases:,} new)", icon=":material/arrow_forward:")

st.subheader("Latest blocked transfers")
latest = df[df["blocked"]].sort_values(["step", "row_id"], ascending=False).head(8)
st.dataframe(
    latest[["step", "type", "amount", "nameOrig", "nameDest", "score"]],
    hide_index=True, width="stretch",
    column_config={"step": "Hour", "type": "Type", "nameOrig": "From", "nameDest": "To",
                   "amount": st.column_config.NumberColumn("Amount", format="%,.2f"),
                   "score": core.risk_column()})
