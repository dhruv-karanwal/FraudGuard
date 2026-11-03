"""Stage 1: exploratory analysis. Prints key facts and saves EDA figures.

Run: python -m src.eda
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import chronological_split, load_paysim


def summarize(df: pd.DataFrame) -> dict:
    fraud = df[config.TARGET] == 1
    by_type = df.groupby("type", observed=True)[config.TARGET].agg(["count", "sum"])
    orig_counts = df["nameOrig"].value_counts()
    dest_counts = df["nameDest"].value_counts()
    flagged = df["isFlaggedFraud"] == 1
    return {
        "rows": int(len(df)),
        "fraud_rows": int(fraud.sum()),
        "fraud_rate": float(fraud.mean()),
        "steps": [int(df["step"].min()), int(df["step"].max())],
        "missing_values": int(df.isna().sum().sum()),
        "rows_by_type": {k: int(v) for k, v in by_type["count"].items()},
        "fraud_by_type": {k: int(v) for k, v in by_type["sum"].items()},
        "unique_nameOrig": int(orig_counts.size),
        "nameOrig_with_history": int((orig_counts > 1).sum()),
        "max_txn_per_nameOrig": int(orig_counts.max()),
        "unique_nameDest": int(dest_counts.size),
        "isFlaggedFraud_rows": int(flagged.sum()),
        "isFlaggedFraud_precision": float(df.loc[flagged, config.TARGET].mean()) if flagged.any() else None,
    }


def plot_class_distribution(df: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df[config.TARGET].value_counts().sort_index()
    axes[0].bar(["Legit", "Fraud"], counts.values, color=["#4C72B0", "#C44E52"])
    axes[0].set_yscale("log")
    axes[0].set_title("Class distribution (log scale)")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n({v / counts.sum():.3%})", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(1, counts.max() * 8)

    ct = df.groupby(["type", config.TARGET], observed=True).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#C44E52"], logy=True)
    axes[1].set_title("Transactions by type and class (log scale)")
    axes[1].set_xlabel("")
    axes[1].legend(["Legit", "Fraud"])
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fraud_over_time(df: pd.DataFrame, bounds, path) -> None:
    per_step = df.groupby("step")[config.TARGET].agg(["count", "sum"])
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(per_step.index, per_step["count"], color="#4C72B0", lw=0.8, label="All transactions")
    ax1.set_ylabel("Transactions per step")
    ax1.set_xlabel("Step (hour)")
    ax2 = ax1.twinx()
    ax2.plot(per_step.index, per_step["sum"], color="#C44E52", lw=0.8, label="Fraud")
    ax2.set_ylabel("Fraud per step")
    top = ax1.get_ylim()[1]
    for b, name, y in zip(bounds, ("train | val", "val | test"), (0.95, 0.85)):
        ax1.axvline(b + 0.5, color="k", ls="--", lw=1)
        ax1.text(b + 0.5, top * y, f" {name} ", fontsize=9, ha="center",
                 bbox=dict(fc="white", ec="none", alpha=0.8))
    ax1.set_title("Volume and fraud over time with chronological split boundaries")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_amount(df: pd.DataFrame, path) -> None:
    sub = df[df["type"].isin(config.FRAUD_TYPES)]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 8, 60)
    for label, color, name in ((0, "#4C72B0", "Legit"), (1, "#C44E52", "Fraud")):
        vals = np.log10(sub.loc[sub[config.TARGET] == label, "amount"] + 1)
        ax.hist(vals, bins=bins, density=True, alpha=0.6, color=color, label=name)
    ax.set_xlabel("log10(amount + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Amount distribution (TRANSFER + CASH_OUT)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    df = load_paysim()
    df, bounds = chronological_split(df)
    stats = summarize(df)
    scored = df[df["type"].isin(config.FRAUD_TYPES)]
    stats["split_boundaries"] = {"last_train_step": bounds[0], "last_val_step": bounds[1]}
    stats["split_scored"] = {
        s: {"rows": int((scored["split"] == s).sum()),
            "fraud": int(scored.loc[scored["split"] == s, config.TARGET].sum())}
        for s in ("train", "val", "test")
    }
    print(json.dumps(stats, indent=2))
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(stats, indent=2))

    plot_class_distribution(df, config.FIGURES_DIR / "class_distribution.png")
    plot_fraud_over_time(df, bounds, config.FIGURES_DIR / "fraud_over_time.png")
    plot_amount(df, config.FIGURES_DIR / "amount_distribution.png")
    print("Figures written to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
