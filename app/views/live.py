"""Live monitor: replay the monitoring window hour by hour as if transfers were arriving now."""
import altair as alt
import streamlit as st

from app import core

st.title("Live monitor")
df = core.transactions()
if df is None:
    st.info("No scored transactions yet.")
    st.stop()

df["blocked"] = df["tier"] == "BLOCK"
df["caught"] = df["blocked"] & (df["isFraud"] == 1)
hourly = df.groupby("step").agg(scored=("row_id", "size"), alerts=("blocked", "sum"), caught=("caught", "sum"),
                                fraud=("isFraud", "sum")).reset_index()
hours = hourly["step"].tolist()

ss = st.session_state
ss.setdefault("live_pos", 0)
ss.setdefault("live_running", False)

st.caption("Replays the test period as a live feed: each tick, the next simulated hour of transfers arrives "
           "and is scored by the model. Use Pause to stop updates at any time.")
c = st.columns([1, 1, 1.4, 3])
if c[0].button("Pause" if ss.live_running else ("Resume" if ss.live_pos else "Start"), type="primary",
               icon=":material/pause:" if ss.live_running else ":material/play_arrow:", width="stretch"):
    ss.live_running = not ss.live_running and ss.live_pos < len(hours)
    st.rerun()
if c[1].button("Reset", icon=":material/restart_alt:", width="stretch"):
    ss.live_pos, ss.live_running = 0, False
    st.rerun()
speed = c[2].selectbox("Speed", [1, 3, 6, 12], format_func=lambda s: f"{s} h per second", index=1)


@st.fragment(run_every=1.0 if ss.live_running else None)
def board():
    pos = ss.live_pos
    if ss.live_running and pos < len(hours):
        new_pos = min(pos + speed, len(hours))
        caught_now = int(hourly["caught"].iloc[pos:new_pos].sum())
        if caught_now:
            st.toast(f"{caught_now} fraudulent transfer(s) blocked in hours {hours[pos]}–{hours[new_pos - 1]}",
                     icon=":material/gpp_bad:")
        ss.live_pos = pos = new_pos
        if pos >= len(hours):
            ss.live_running = False
            st.rerun()  # full rerun stops the timer and flips the button back
    seen = hourly.iloc[:pos]
    now = f"h{hours[pos - 1]}" if pos else "–"
    done = df[df["step"] <= hours[pos - 1]] if pos else df.iloc[:0]
    protected = float(done.loc[done["caught"], "amount"].sum())

    k = st.columns(5)
    # Compact values so five cards fit narrow screens; exact figures are in the tooltips.
    k[0].metric("Hour", now, border=True)
    k[1].metric("Scored", core.fmt_money(seen["scored"].sum()), border=True,
                help=f"{int(seen['scored'].sum()):,} transfers scored")
    k[2].metric("Alerts", f"{int(seen['alerts'].sum()):,}", border=True)
    k[3].metric("Blocked", f"{int(seen['caught'].sum()):,}", border=True,
                help=f"{int(seen['caught'].sum()):,} of {int(seen['fraud'].sum()):,} fraudulent transfers blocked")
    k[4].metric("Protected", core.fmt_money(protected), border=True, help="Amount of fraud blocked")
    st.progress(pos / len(hours), text=f"{pos} of {len(hours)} hours replayed")

    left, right = st.columns([2, 1])
    with left:
        if len(seen):
            long = seen.melt("step", ["alerts", "fraud"], var_name="Series", value_name="Count")
            chart = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
                x=alt.X("step:Q", title="Simulated hour", scale=alt.Scale(domain=[hours[0], hours[-1]])),
                y=alt.Y("Count:Q", title="Transfers per hour"),
                color=alt.Color("Series:N", scale=alt.Scale(domain=["alerts", "fraud"], range=["#38BDF8", "#94A3B8"])),
                strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["alerts", "fraud"],
                                                                      range=[[1, 0], [4, 3]])))
            st.altair_chart(chart.properties(height=280), width="stretch")
        else:
            st.info("Press Start to begin the replay.")
    with right:
        st.markdown("**Latest blocks**")
        feed = done[done["blocked"]].sort_values(["step", "row_id"], ascending=False).head(8)
        for _, r in feed.iterrows():
            st.markdown(f"{core.tier_chip('BLOCK')} h{int(r['step'])} · {r['type']} · {core.fmt_money(r['amount'])}",
                        unsafe_allow_html=True)
        if feed.empty:
            st.markdown('<span class="fg-muted">No blocks yet.</span>', unsafe_allow_html=True)


board()
