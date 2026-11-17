"""Draw the documentation diagrams and copy result figures into docs/images.

Run: python docs/make_diagrams.py   (after src.eda and src.evaluate)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

OUT = ROOT / "docs" / "images"

C_DATA, C_PROC, C_MODEL, C_EVAL, C_APP, C_STORE = (
    "#DCE6F2", "#E2EFDA", "#F2DCDB", "#FFF2CC", "#E4DFEC", "#F2F2F2")


def box(ax, x, y, w, h, title, body="", color="#fff", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=color, ec="#333", lw=1))
    if body:
        ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=fs + 0.5, weight="bold")
        ax.text(x + w / 2, y + h - 0.05, body, ha="center", va="top", fontsize=fs - 0.5, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold")


def arrow(ax, p, q, text="", style="-|>", color="#333", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=1.2, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + 0.005, (p[1] + q[1]) / 2, text, fontsize=7.5, color="#555",
                ha="left", va="center")


def system_architecture(path):
    fig, ax = plt.subplots(figsize=(14, 9.6))
    ax.set(xlim=(0, 1), ylim=(0, 1.04))
    ax.axis("off")
    ax.text(0.5, 1.035, "FraudGuard system architecture", ha="center", va="top", fontsize=15, weight="bold")

    # Layer labels
    layers = [(0.85, "1. Data"), (0.66, "2. Processing"), (0.44, "3. Modelling"),
              (0.235, "4. Evaluation"), (0.045, "5. Serving")]
    for y, name in layers:
        ax.text(0.01, y + 0.06, name, fontsize=10, weight="bold", color="#444", rotation=90, va="center")

    # 1. Data
    box(ax, 0.06, 0.85, 0.2, 0.11, "PaySim CSV", "6,362,620 rows, 11 columns\ndata/raw/paysim.csv", C_DATA)
    box(ax, 0.32, 0.85, 0.2, 0.11, "Loader (src/data.py)", "explicit dtypes, row_id\nparquet cache", C_DATA)
    box(ax, 0.58, 0.85, 0.2, 0.11, "EDA (src/eda.py)", "class balance, fraud over time\nnameOrig repeat analysis", C_DATA)
    arrow(ax, (0.26, 0.905), (0.32, 0.905))
    arrow(ax, (0.52, 0.905), (0.58, 0.905))

    # 2. Processing
    box(ax, 0.06, 0.64, 0.2, 0.13, "Chronological split", "by whole steps, 70/15/15\ntrain 1-323 | val 324-377\ntest 378-743", C_PROC)
    box(ax, 0.32, 0.64, 0.2, 0.13, "Feature engineering", "14 row-local features\nbalance errors, drained flag\nlog amounts, cyclic hour", C_PROC)
    box(ax, 0.58, 0.64, 0.2, 0.13, "Sequence builder", "last 5 txns per nameOrig\ngrouped by (split, account)\nleft-padded + mask", C_PROC)
    box(ax, 0.83, 0.64, 0.15, 0.13, "Scalers", "StandardScaler\nfit on train only", C_PROC)
    arrow(ax, (0.42, 0.85), (0.16, 0.77), rad=0.0)
    arrow(ax, (0.26, 0.705), (0.32, 0.705))
    arrow(ax, (0.52, 0.705), (0.58, 0.705))
    arrow(ax, (0.78, 0.705), (0.83, 0.705))

    # store
    box(ax, 0.58, 0.535, 0.4, 0.07, "data/processed/<tag>/  {split}_{tab, seq, mask, y}.npy  + meta", color=C_STORE, fs=8.5)
    arrow(ax, (0.905, 0.64), (0.85, 0.605))

    # 3. Modelling
    box(ax, 0.06, 0.38, 0.4, 0.13, "Baselines (src/baselines.py)",
        "LogReg  |  RandomForest  |  XGBoost  |  DNN-only\nsame split + tabular features, class weights", C_MODEL)
    box(ax, 0.52, 0.38, 0.46, 0.13, "Hybrid model (src/models.py, src/train.py)",
        "DNN branch (transaction)  +  GRU/LSTM branch (history)\nmasked temporal attention, attention fusion, sigmoid\n"
        "balanced class weights, early stop on val PR-AUC", C_MODEL)
    arrow(ax, (0.68, 0.535), (0.26, 0.51), rad=0.05)
    arrow(ax, (0.78, 0.535), (0.75, 0.51))

    box(ax, 0.06, 0.305, 0.92, 0.05, "models/<tag>/  *.joblib  *.keras  scalers  *_val_scores.npy  training history",
        color=C_STORE, fs=8.5)
    arrow(ax, (0.26, 0.38), (0.26, 0.355))
    arrow(ax, (0.75, 0.38), (0.75, 0.355))

    # 4. Evaluation
    box(ax, 0.06, 0.16, 0.28, 0.12, "Threshold selection", "F1-optimal threshold\nfrom validation scores only", C_EVAL)
    box(ax, 0.38, 0.16, 0.28, 0.12, "One-shot test scoring", "all 6 models scored once\nmarker file blocks re-runs", C_EVAL)
    box(ax, 0.70, 0.16, 0.28, 0.12, "Metrics + figures", "P, R, F1, ROC-AUC, PR-AUC\nconfusion matrices, curves", C_EVAL)
    arrow(ax, (0.2, 0.305), (0.2, 0.28))
    arrow(ax, (0.34, 0.22), (0.38, 0.22))
    arrow(ax, (0.66, 0.22), (0.70, 0.22))

    # 5. Serving
    box(ax, 0.06, 0.02, 0.28, 0.11, "Predictor (src/inference.py)",
        "raw fields -> same features\n-> probability + attention", C_APP)
    box(ax, 0.38, 0.02, 0.6, 0.11, "FraudGuard console (app/)",
        "Overview | Alert queue | Investigate | Live monitor | Check | Model\nBLOCK / REVIEW / APPROVE decisions", C_APP)
    arrow(ax, (0.2, 0.16), (0.2, 0.13), text="model + threshold")
    arrow(ax, (0.34, 0.075), (0.38, 0.075))
    arrow(ax, (0.84, 0.16), (0.84, 0.13), text="reports/ figures, metrics")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def workflow(path):
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.set(xlim=(0, 1), ylim=(-0.12, 1))
    ax.axis("off")
    ax.text(0.5, 0.97, "Development workflow: outer loop over stages, inner loop per stage",
            ha="center", va="top", fontsize=14, weight="bold")
    stages = ["1. Data\n+ EDA", "2. Features\n+ sequences", "3. Baselines", "4. Hybrid\nmodel",
              "5. Evaluation\n+ figures", "6. Streamlit\napp", "7. README\n+ docs"]
    w, gap, y = 0.115, 0.0233, 0.6
    xs = [0.02 + i * (w + gap) for i in range(len(stages))]
    for x, s in zip(xs, stages):
        box(ax, x, y, w, 0.2, s, color=C_PROC, fs=10)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y + 0.1), (b, y + 0.1))

    # inner loop
    cx, cy = 0.5, 0.25
    steps = [("Implement", cx - 0.3), ("Run", cx - 0.1), ("Check\n(tests / metrics /\nsanity checks)", cx + 0.1),
             ("Pass?\nupdate guide\n+ commit", cx + 0.3)]
    for name, x in steps:
        box(ax, x - 0.08, cy - 0.09, 0.16, 0.18, name, color=C_EVAL if "Pass" not in name else C_APP, fs=9.5)
    for (_, a), (_, b) in zip(steps[:-1], steps[1:]):
        arrow(ax, (a + 0.08, cy), (b - 0.08, cy))
    arrow(ax, (cx + 0.1, cy - 0.09), (cx - 0.3, cy - 0.09), text="", color="#C44E52", rad=-0.35)
    ax.text(cx - 0.1, -0.06, "check fails -> fix and repeat", color="#C44E52", ha="center", fontsize=9)
    arrow(ax, (xs[3] + w / 2, y), (cx, cy + 0.09), color="#888", ls="--")
    ax.text(xs[3] + w / 2 + 0.01, 0.47, "every stage runs\nthis inner loop", fontsize=8.5, color="#666")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def data_flow(path):
    """Tensor shapes through the pipeline for the full dataset."""
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.5, 0.98, "Data flow and shapes (full dataset)", ha="center", va="top", fontsize=13, weight="bold")
    items = [
        ("Raw CSV", "6,362,620 x 11"),
        ("Scored rows", "TRANSFER + CASH_OUT\n2,770,409"),
        ("Split", "train 1,951,895\nval 410,437\ntest 408,077"),
        ("Tabular input", "(n, 14)"),
        ("Sequence input", "(n, 5, 13) + mask (n, 5)"),
        ("Output", "P(fraud) in [0, 1]\n-> threshold from val"),
    ]
    w = 0.145
    for i, (t, b) in enumerate(items):
        x = 0.01 + i * 0.165
        box(ax, x, 0.12, w, 0.68, "", color=[C_DATA, C_DATA, C_PROC, C_PROC, C_PROC, C_MODEL][i])
        ax.text(x + w / 2, 0.70, t, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, 0.40, b, ha="center", va="center", fontsize=9.5, linespacing=1.5)
        if i:
            arrow(ax, (x - 0.02, 0.46), (x, 0.46))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    system_architecture(OUT / "system_architecture.png")
    workflow(OUT / "workflow.png")
    data_flow(OUT / "data_flow.png")
    for f in sorted(config.FIGURES_DIR.glob("*.png")):
        shutil.copy2(f, OUT / f.name)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
