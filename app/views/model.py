"""Model & reports: cost-based threshold explorer, model card, test metrics and figures."""
import altair as alt
import pandas as pd
import streamlit as st

from app import core
from src import config

LABELS = {"logreg": "LogReg", "random_forest": "RandomForest", "xgboost": "XGBoost",
          "dnn_only": "DNN-only", "hybrid_gru": "Hybrid (DNN+GRU)", "hybrid_lstm": "Hybrid (DNN+LSTM)"}

st.title("Model & reports")
tabs = st.tabs(["Threshold & cost", "Model card", "Test results", "Figures"])

with tabs[0]:
    val = core.transactions("val")
    if val is None:
        st.info("Validation scores not available.")
    else:
        st.caption("Explore the trade-off on the **validation** period (hours 324–377), so the test set stays "
                   "untouched. A missed fraud costs its amount; a false alarm costs analyst time and customer "
                   "friction, which you set here.")
        fa_cost = st.slider("Cost of one false alarm", 0, 50_000, 2_000, step=500)
        curve = core.cost_curve(val, fa_cost)
        best = curve.loc[curve["total_cost"].idxmin()]
        cur_t = core.threshold()
        cur_m = core.money(val, cur_t)
        cur_cost = cur_m["lost"] + cur_m["false_alarms"] * fa_cost

        saving = cur_cost - best["total_cost"]
        c = st.columns(4)
        c[0].metric("Cost-optimal threshold", f"{best['threshold']:.4f}", border=True)
        c[1].metric("Total cost at optimum", core.fmt_money(best["total_cost"]), border=True,
                    delta=f"-{core.fmt_money(saving)} vs deployed" if saving > 0 else None, delta_color="inverse",
                    help=None if saving > 0 else "Same as the deployed threshold.")
        c[2].metric("Missed · false alarms", f"{int(best['missed'])} · {int(best['false_alarms'])}", border=True,
                    help=f"Recall {best['recall']:.2%}, precision {best['precision']:.2%}")
        c[3].metric("Deployed threshold", f"{cur_t:.4f}", border=True,
                    help=f"F1-tuned on validation. Total cost there: {core.fmt_money(cur_cost)}")

        # Trade-off curve: every point is one threshold; lower-left is better.
        pts = curve[curve["missed_amount"] > 0].assign(fa=lambda d: d["false_alarms"] + 1)
        marks = pd.DataFrame({"fa": [best["false_alarms"] + 1, cur_m["false_alarms"] + 1],
                              "missed_amount": [max(best["missed_amount"], 1), max(cur_m["lost"], 1)],
                              "label": ["Cost-optimal", "Deployed"]})
        line = alt.Chart(pts).mark_line(color="#38BDF8", strokeWidth=2).encode(
            x=alt.X("fa:Q", scale=alt.Scale(type="log"), title="False alarms + 1 (log)"),
            y=alt.Y("missed_amount:Q", scale=alt.Scale(type="log"), title="Missed fraud amount (log)"),
            tooltip=[alt.Tooltip("threshold", format=".6f"), alt.Tooltip("false_alarms", title="False alarms"),
                     alt.Tooltip("missed", title="Missed frauds"),
                     alt.Tooltip("missed_amount", title="Missed amount", format=",.0f")])
        dots = alt.Chart(marks).mark_point(size=160, filled=True).encode(
            x="fa:Q", y="missed_amount:Q",
            shape=alt.Shape("label:N", scale=alt.Scale(domain=["Cost-optimal", "Deployed"],
                                                       range=["diamond", "circle"]), title=None),
            color=alt.Color("label:N", scale=alt.Scale(domain=["Cost-optimal", "Deployed"],
                                                       range=["#22C55E", "#F59E0B"]), title=None))
        st.altair_chart((line + dots).properties(height=320), width="stretch")
        st.caption("Each point on the line is one threshold: moving left means fewer false alarms, moving down "
                   "means less fraud slips through. Raise the false-alarm cost to watch the optimum move left. "
                   "The deployed threshold stays the F1-tuned one reported in the project.")

with tabs[1]:
    meta = core.predictor().meta
    b = meta["split_boundaries"]
    st.markdown(f"""
**Model**: hybrid DNN + GRU with masked temporal attention and attention fusion (`{core.MODEL}`, data `{core.TAG}`).

**Intended use**: score TRANSFER and CASH_OUT mobile-money transactions in real time; block high-risk
transfers and queue medium-risk ones for analyst review.

**Training data**: PaySim, a simulation of one month of mobile-money logs. Chronological split: train hours
1–{b['last_train_step']}, validation {b['last_train_step'] + 1}–{b['last_val_step']}, test
{b['last_val_step'] + 1}–743.

**Inputs**: {len(meta['tab_features'])} transaction features and the last {meta['seq_len']} transfers of the
origin account ({len(meta['seq_features'])} features each).

**Decision rule**: BLOCK at probability ≥ {core.threshold():.4f} (F1-optimal on validation), REVIEW from
{core.REVIEW_FLOOR:.2f}, otherwise APPROVE.

**Limitations**: synthetic data with a very clean fraud pattern (accounts drained in full), so real-world
performance will be lower. Almost no PaySim account has more than one transfer, so the history branch carries
little weight. Fraud prevalence rises over time, so the threshold should be re-checked as data drifts.
""")

with tabs[2]:
    res = core.read_json(str(config.REPORTS_DIR / f"{core.TAG}_test_metrics.json"))
    if not res:
        st.info("Test metrics not available yet. Run `python -m src.evaluate`.")
    else:
        st.caption("Test period only; each model was scored once and each threshold was chosen on validation.")
        rows = []
        for n, m in res.items():
            (tn, fp), (fn, tp) = m["confusion_matrix"]
            rows.append({"model": LABELS.get(n, n), "precision": m["precision"], "recall": m["recall"],
                         "f1": m["f1"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "threshold": m["threshold"],
                         "TN": tn, "FP": fp, "FN": fn, "TP": tp})
        table = pd.DataFrame(rows).set_index("model")
        st.dataframe(table.style.format({c: "{:.4f}" for c in ["precision", "recall", "f1", "roc_auc", "pr_auc",
                                                                "threshold"]})
                     .highlight_max(subset=["precision", "recall", "f1", "roc_auc", "pr_auc"],
                                    props="font-weight: bold; color: #22C55E"),
                     width="stretch")
        core.figure("model_comparison.png")

with tabs[3]:
    sub = st.segmented_control("Figure", ["Confusion matrices", "ROC / PR", "Training", "Architecture", "Data"],
                               default="Confusion matrices")
    if sub == "Confusion matrices":
        core.figure("confusion_matrix_hybrid.png", "Hybrid model; layout [[TN, FP], [FN, TP]]")
        cols = st.columns(2)
        others = ["hybrid_lstm", "dnn_only", "xgboost", "random_forest", "logreg"]
        for i, n in enumerate(others):
            with cols[i % 2]:
                core.figure(f"confusion_matrix_{n}.png")
    elif sub == "ROC / PR":
        core.figure("roc_pr_curves.png")
    elif sub == "Training":
        core.figure("training_curves.png")
    elif sub == "Architecture":
        core.figure("architecture.png")
        summ = config.REPORTS_DIR / "hybrid_model_summary.txt"
        if summ.exists():
            st.code(summ.read_text(encoding="utf-8"), language="text")
    elif sub == "Data":
        core.figure("class_distribution.png")
        core.figure("amount_distribution.png")
        core.figure("fraud_over_time.png")
