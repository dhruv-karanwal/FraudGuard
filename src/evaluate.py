"""Stage 5: single test-set evaluation of every model, plus report figures.

Run: python -m src.evaluate [--dev] [--force]

Thresholds come from validation scores saved during training. The test set is
scored once; a marker file blocks accidental re-runs (use --force knowingly).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config
from src.metrics import best_f1_threshold, evaluate
from src.prepare import load_split

MODELS = ["logreg", "random_forest", "xgboost", "dnn_only", "hybrid_gru"]
LABELS = {"logreg": "LogReg", "random_forest": "RandomForest", "xgboost": "XGBoost",
          "dnn_only": "DNN-only", "hybrid_gru": "Hybrid (DNN+GRU)", "hybrid_lstm": "Hybrid (DNN+LSTM)"}


def score(name: str, model_dir, d: dict) -> np.ndarray:
    if name.startswith("hybrid") or name == "dnn_only":
        import keras
        import src.models  # noqa: F401  (registers custom layers)
        model = keras.models.load_model(model_dir / f"{name}.keras")
        x = d["tab"] if name == "dnn_only" else {"tab": d["tab"], "seq": d["seq"], "mask": d["mask"]}
        return model.predict(x, batch_size=8192, verbose=0).ravel()
    return joblib.load(model_dir / f"{name}.joblib").predict_proba(d["tab"])[:, 1]


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(cm.max(), 2)))
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 10 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Pred legit", "Pred fraud"])
    ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results, path):
    """Left: metric bars (zoomed y-axis, scores are all near 1). Right: error counts."""
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    names = list(results)
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    lo = min(results[n][m] for n in names for m in metrics)
    for i, n in enumerate(names):
        vals = [results[n][m] for m in metrics]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=LABELS.get(n, n))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)
    ax.set_xticks(x, ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(max(0.0, lo - (1 - lo) * 0.6 - 0.002), 1.0 + (1 - lo) * 0.35 + 0.001)
    ax.set_title("Test metrics (thresholds tuned on validation; y-axis zoomed)")
    ax.legend(ncol=len(names), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07))

    fp = [results[n]["confusion_matrix"][0][1] for n in names]
    fn = [results[n]["confusion_matrix"][1][0] for n in names]
    y = np.arange(len(names))
    ax2.barh(y - 0.2, fp, 0.4, label="False positives", color="#DD8452")
    ax2.barh(y + 0.2, fn, 0.4, label="False negatives (missed fraud)", color="#C44E52")
    for yi, a, b in zip(y, fp, fn):
        ax2.text(a, yi - 0.2, f" {a}", va="center", fontsize=8)
        ax2.text(b, yi + 0.2, f" {b}", va="center", fontsize=8)
    ax2.set_yticks(y, [LABELS.get(n, n) for n in names])
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(fp + fn) * 1.25 + 1)
    ax2.set_title("Test errors")
    ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(y, scores: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for n, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, lw=1.2, label=LABELS.get(n, n))
        axes[1].plot(r, p, lw=1.2, label=LABELS.get(n, n))
    axes[0].set(xscale="log", xlim=(1e-6, 1), ylim=(0.9, 1.002), xlabel="False positive rate (log)",
                ylabel="True positive rate", title="ROC (test, zoomed)")
    axes[1].set(xlim=(0.9, 1.002), ylim=(0.9, 1.002), xlabel="Recall", ylabel="Precision",
                title="Precision-recall (test, zoomed)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(model_dir, names, path):
    hs = {n: json.loads((model_dir / f"{n}_history.json").read_text())
          for n in names if (model_dir / f"{n}_history.json").exists()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n, h in hs.items():
        ep = np.arange(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["loss"], label=f"{LABELS.get(n, n)} train")
        axes[0].plot(ep, h["val_loss"], "--", label=f"{LABELS.get(n, n)} val")
        axes[1].plot(ep, h["pr_auc"], label=f"{LABELS.get(n, n)} train")
        axes[1].plot(ep, h["val_pr_auc"], "--", label=f"{LABELS.get(n, n)} val")
    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE loss", title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="PR-AUC", title="PR-AUC")
    for a in axes:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_architecture(model_dir, path):
    """Block diagram of the hybrid model, annotated from the saved model."""
    import keras
    import src.models  # noqa: F401
    m = keras.models.load_model(model_dir / "hybrid_gru.keras")
    n_tab = m.get_layer("tab").output.shape[-1] if hasattr(m.get_layer("tab"), "output") else "?"
    seq_shape = m.inputs[1].shape
    params = m.count_params()

    boxes = {
        "tab_in": (0.05, 0.78, f"Transaction features\n({n_tab})"),
        "dnn": (0.05, 0.50, "DNN branch\nDense128-BN-ReLU-Drop\nDense64-Drop-Dense32"),
        "seq_in": (0.55, 0.78, f"Last {seq_shape[1]} txns of nameOrig\n({seq_shape[1]} x {seq_shape[2]}) + mask"),
        "gru": (0.55, 0.50, "GRU(32, seq)\nmasked temporal attention\nDense32"),
        "fuse": (0.30, 0.25, "Attention fusion\nsoftmax over [h_tab, h_seq]"),
        "head": (0.30, 0.03, "Dense32-Drop -> Dense1 sigmoid\nP(fraud)"),
    }
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axis("off")
    W, H = 0.38, 0.15
    for key, (x, y, txt) in boxes.items():
        color = "#DCE6F2" if key in ("tab_in", "dnn") else "#F2DCDB" if key in ("seq_in", "gru") else "#E2EFDA"
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.01",
                                                       fc=color, ec="k"))
        ax.text(x + W / 2, y + H / 2, txt, ha="center", va="center", fontsize=9)

    def arrow(a, b):
        xa, ya, _ = boxes[a]
        xb, yb, _ = boxes[b]
        ax.annotate("", xy=(xb + W / 2, yb + H), xytext=(xa + W / 2, ya),
                    arrowprops=dict(arrowstyle="->", lw=1.2))

    for a, b in [("tab_in", "dnn"), ("seq_in", "gru"), ("dnn", "fuse"), ("gru", "fuse"), ("fuse", "head")]:
        arrow(a, b)
    ax.set_title(f"Hybrid FraudGuard model ({params:,} parameters)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    lines = []
    m.summary(print_fn=lambda s, **_: lines.append(s))
    (config.REPORTS_DIR / "hybrid_model_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main(dev: bool, force: bool) -> None:
    tag = "dev" if dev else "full"
    model_dir = config.MODELS_DIR / tag
    marker = model_dir / "test_evaluated.json"
    if marker.exists() and not force:
        raise SystemExit(f"Test set already evaluated for '{tag}' ({marker}). Use --force to re-run.")

    va, te = load_split(tag, "val"), load_split(tag, "test")
    names = [n for n in MODELS + ["hybrid_lstm"]
             if (model_dir / f"{n}.joblib").exists() or (model_dir / f"{n}.keras").exists()]

    results, test_scores = {}, {}
    for n in names:
        s_val = np.load(model_dir / f"{n}_val_scores.npy")
        thr = best_f1_threshold(va["y"], s_val)
        s_test = score(n, model_dir, te)
        np.save(model_dir / f"{n}_test_scores.npy", s_test)
        test_scores[n] = s_test
        results[n] = evaluate(te["y"], s_test, thr)
        print(f"{n:15s} thr={thr:.4f} P={results[n]['precision']:.4f} R={results[n]['recall']:.4f} "
              f"F1={results[n]['f1']:.4f} ROC={results[n]['roc_auc']:.4f} PR={results[n]['pr_auc']:.4f} "
              f"CM={results[n]['confusion_matrix']}")

    out = config.REPORTS_DIR / f"{tag}_test_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    marker.write_text(json.dumps({"models": names}, indent=2))
    make_figures(tag, results, te["y"], test_scores)
    print("wrote", out, "and figures to", config.FIGURES_DIR)


def redraw(dev: bool) -> None:
    """Redraw figures from saved test scores/metrics; does not re-score the test set."""
    tag = "dev" if dev else "full"
    model_dir = config.MODELS_DIR / tag
    results = json.loads((config.REPORTS_DIR / f"{tag}_test_metrics.json").read_text())
    y = np.load(config.PROCESSED_DIR / tag / "test_y.npy")
    scores = {n: np.load(model_dir / f"{n}_test_scores.npy") for n in results}
    make_figures(tag, results, y, scores)
    print("redrew figures in", config.FIGURES_DIR)


def make_figures(tag: str, results: dict, y_test: np.ndarray, test_scores: dict) -> None:
    model_dir = config.MODELS_DIR / tag
    names = list(results)
    fig_dir = config.FIGURES_DIR
    best = "hybrid_gru" if "hybrid_gru" in results else names[-1]
    plot_confusion(results[best]["confusion_matrix"], f"{LABELS.get(best, best)}: test confusion matrix",
                   fig_dir / "confusion_matrix_hybrid.png")
    for n in names:
        if n != best:
            plot_confusion(results[n]["confusion_matrix"], f"{LABELS.get(n, n)}: test",
                           fig_dir / f"confusion_matrix_{n}.png")
    plot_comparison(results, fig_dir / "model_comparison.png")
    plot_curves(y_test, test_scores, fig_dir / "roc_pr_curves.png")
    plot_history(model_dir, ["hybrid_gru", "dnn_only"], fig_dir / "training_curves.png")
    if (model_dir / "hybrid_gru.keras").exists():
        plot_architecture(model_dir, fig_dir / "architecture.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw figures from saved test scores (no re-scoring)")
    args = ap.parse_args()
    redraw(args.dev) if args.figures_only else main(args.dev, args.force)
