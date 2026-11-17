"""Shared data, business rules and UI helpers for the FraudGuard console."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.inference import Predictor, available_tag

TAG = available_tag()
MODEL = "hybrid_gru"
ASSETS = Path(__file__).resolve().parent / "assets"

# Decision tiers. BLOCK uses the validation-tuned threshold; REVIEW is a fixed band below it.
REVIEW_FLOOR = 0.5  # shortcut: fixed review band [0.5, threshold); make it a setting if analysts need to tune it
TIERS = ("BLOCK", "REVIEW", "APPROVE")
TIER_COLOR = {"BLOCK": "#EF4444", "REVIEW": "#F59E0B", "APPROVE": "#22C55E"}
STATUSES = ("New", "Investigating", "Escalated", "Confirmed fraud", "False positive")


# ---------- data ----------
@st.cache_resource(show_spinner="Loading fraud model...")
def predictor() -> Predictor:
    return Predictor(TAG, MODEL)


@st.cache_data
def read_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner="Loading transactions...")
def transactions(split: str = "test") -> pd.DataFrame | None:
    """Scored TRANSFER/CASH_OUT rows of one split with model score and tier."""
    meta = config.PROCESSED_DIR / TAG / f"{split}_meta.parquet"
    scores = config.MODELS_DIR / TAG / f"{MODEL}_{split}_scores.npy"
    if not (meta.exists() and scores.exists()):
        return None
    df = pd.read_parquet(meta)
    df["type"] = df["type"].astype(str)
    df["score"] = np.load(scores)
    df["tier"] = tier(df["score"].to_numpy(), threshold())
    return df


def threshold() -> float:
    return predictor().threshold


def tier(scores, thr: float) -> np.ndarray:
    s = np.asarray(scores)
    return np.where(s >= thr, "BLOCK", np.where(s >= REVIEW_FLOOR, "REVIEW", "APPROVE"))


# ---------- business rules ----------
def signals(row: dict, large_amount: float = np.inf) -> list[str]:
    """Human-readable risk signals for one transaction (descriptive, not the model itself)."""
    amt, ob, nb = row["amount"], row["oldbalanceOrg"], row["newbalanceOrig"]
    obd, nbd = row["oldbalanceDest"], row["newbalanceDest"]
    out = []
    if ob > 0 and abs(amt - ob) < 0.01:
        out.append("Origin account fully drained")
    elif ob > 0 and nb == 0:
        out.append("Origin left with zero balance")
    if obd == 0 and nbd == 0:
        out.append("Destination balance unchanged")
    elif abs(obd + amt - nbd) > 0.01:
        out.append("Destination balance does not add up")
    if amt >= large_amount:
        out.append("Unusually large amount")
    if row["type"] == "TRANSFER":
        out.append("Transfer to another customer")
    return out


def money(df: pd.DataFrame, thr: float) -> dict:
    """Outcome of blocking every transaction with score >= thr."""
    flagged = df["score"].to_numpy() >= thr
    fraud = df["isFraud"].to_numpy() == 1
    amt = df["amount"].to_numpy()
    return {
        "scored": int(len(df)),
        "alerts": int(flagged.sum()),
        "fraud": int(fraud.sum()),
        "caught": int((flagged & fraud).sum()),
        "missed": int((~flagged & fraud).sum()),
        "false_alarms": int((flagged & ~fraud).sum()),
        "protected": float(amt[flagged & fraud].sum()),
        "lost": float(amt[~flagged & fraud].sum()),
    }


def cost_curve(df: pd.DataFrame, false_alarm_cost: float, min_threshold: float = 0.05) -> pd.DataFrame:
    """Exact cost (missed fraud amount + false alarms x cost) at every distinct score threshold.

    Sorting once and taking cumulative sums gives the outcome of `score >= t` for each score t,
    so the minimum is the true optimum, not a grid approximation.
    """
    s = df["score"].to_numpy()
    order = np.argsort(-s, kind="stable")
    s, fraud = s[order], df["isFraud"].to_numpy()[order] == 1
    amt = df["amount"].to_numpy()[order]
    caught = np.cumsum(fraud)
    alerts = np.arange(1, len(s) + 1)
    lost = np.maximum(amt[fraud].sum() - np.cumsum(amt * fraud), 0).round(2)  # round off float residue
    last_of_tie = np.r_[s[1:] != s[:-1], True] & (s >= min_threshold)
    fp = alerts - caught
    out = pd.DataFrame({"threshold": s, "missed_amount": lost, "false_alarms": fp,
                        "missed": fraud.sum() - caught, "recall": caught / max(fraud.sum(), 1),
                        "precision": caught / alerts, "total_cost": lost + fp * false_alarm_cost})
    return out[last_of_tie].reset_index(drop=True)


# ---------- case store (session only) ----------
def cases() -> dict:
    # shortcut: cases live in this browser session only; use a database for real multi-analyst use
    return st.session_state.setdefault("cases", {})


def case(row_id: int) -> dict:
    return cases().get(int(row_id), {"status": "New", "note": ""})


def set_case(row_id: int, status: str, note: str) -> None:
    cases()[int(row_id)] = {"status": status, "note": note}


def with_status(df: pd.DataFrame) -> pd.DataFrame:
    store = cases()
    out = df.copy()
    out["status"] = [store.get(int(r), {}).get("status", "New") for r in out["row_id"]]
    return out


# ---------- formatting / UI ----------
def fmt_money(x: float) -> str:
    """PaySim does not state a currency, so amounts are shown as plain units."""
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:,.2f}{suf}"
    return f"{x:,.0f}"


def chip(text: str, color: str = "#334155", fg: str = "#F8FAFC") -> str:
    return f'<span class="fg-chip" style="background:{color};color:{fg}">{text}</span>'


def tier_chip(t: str) -> str:
    return chip(t, TIER_COLOR[t], "#020617")


CSS = """
<style>
.fg-status{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding:8px 14px;margin:-8px 0 14px;
  border:1px solid #334155;border-radius:10px;background:#0E1223;font-size:0.86rem;color:#94A3B8}
.fg-status b{color:#F8FAFC;font-weight:600}
.fg-live{color:#22C55E;font-weight:700;letter-spacing:.04em}
.fg-live::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#22C55E;
  margin-right:6px;vertical-align:middle;animation:fg-pulse 2s infinite}
@keyframes fg-pulse{0%{opacity:1}50%{opacity:.35}100%{opacity:1}}
@media (prefers-reduced-motion: reduce){.fg-live::before{animation:none}}
.fg-chip{display:inline-block;padding:3px 10px;margin:2px 6px 2px 0;border-radius:999px;font-size:0.82rem;
  font-weight:600;line-height:1.4;white-space:nowrap}
.fg-decision{border-radius:12px;padding:18px 22px;border:2px solid;margin-bottom:8px}
.fg-decision .label{font-size:2rem;font-weight:700;letter-spacing:.04em}
.fg-decision .sub{color:#CBD5E1;font-size:0.92rem}
.fg-muted{color:#94A3B8;font-size:0.9rem}
</style>
"""


def status_bar() -> None:
    p = predictor()
    df = transactions()
    n = f"{len(df):,}" if df is not None else "0"
    b = p.meta["split_boundaries"]
    st.markdown(
        f'<div class="fg-status" role="status"><span class="fg-live">LIVE</span>'
        f'<span>Model <b>{MODEL}</b> ({TAG})</span>'
        f'<span>Block threshold <b>{p.threshold:.4f}</b></span>'
        f'<span>Review band <b>{REVIEW_FLOOR:.2f}–{p.threshold:.4f}</b></span>'
        f'<span>Monitoring window <b>hours {b["last_val_step"] + 1}–743</b></span>'
        f'<span>Scored <b>{n}</b> transfers</span></div>',
        unsafe_allow_html=True)


def decision_card(t: str, prob: float, latency_ms: float | None = None) -> None:
    verb = {"BLOCK": "Transaction blocked", "REVIEW": "Sent to analyst review", "APPROVE": "Transaction approved"}[t]
    lat = f" · scored in {latency_ms:.0f} ms" if latency_ms is not None else ""
    st.markdown(
        f'<div class="fg-decision" style="border-color:{TIER_COLOR[t]};background:{TIER_COLOR[t]}1A">'
        f'<div class="label" style="color:{TIER_COLOR[t]}">{t}</div>'
        f'<div class="sub">{verb} · fraud probability <b>{prob:.2%}</b>{lat}</div></div>',
        unsafe_allow_html=True)


def risk_column():
    """Risk score bar, red because higher means more dangerous."""
    return st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.4f", color=TIER_COLOR["BLOCK"])


def figure(name: str, caption: str | None = None) -> None:
    p = config.FIGURES_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not generated yet. Run `python -m src.evaluate`.")
