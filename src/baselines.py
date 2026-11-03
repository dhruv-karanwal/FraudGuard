"""Stage 3: baselines on the same chronological split and tabular features.

Run: python -m src.baselines [--dev]

Trains LogReg, RandomForest, XGBoost and a DNN-only model on train, picks the
F1-optimal threshold on validation, and reports validation metrics. The test
set is not touched here (see src.evaluate).
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import config
from src.metrics import best_f1_threshold, class_weights, evaluate
from src.prepare import load_split
from src.training import fit_keras, set_seed


def sklearn_models(y_train: np.ndarray) -> dict:
    spw = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    return {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=config.SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.SEED),
        "xgboost": XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr",
            early_stopping_rounds=50, n_jobs=-1, random_state=config.SEED),
    }


def main(dev: bool, only: list[str] | None = None) -> None:
    set_seed()
    tag = "dev" if dev else "full"
    model_dir = config.MODELS_DIR / tag
    tr, va = load_split(tag, "train"), load_split(tag, "val")
    results_path = config.REPORTS_DIR / f"{tag}_val_metrics.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for name, clf in sklearn_models(tr["y"]).items():
        if only and name not in only:
            continue
        t0 = time.time()
        fit_kw = {"eval_set": [(va["tab"], va["y"])], "verbose": False} if name == "xgboost" else {}
        clf.fit(tr["tab"], tr["y"], **fit_kw)
        s_val = clf.predict_proba(va["tab"])[:, 1]
        joblib.dump(clf, model_dir / f"{name}.joblib")
        np.save(model_dir / f"{name}_val_scores.npy", s_val)
        results[name] = evaluate(va["y"], s_val, best_f1_threshold(va["y"], s_val))
        results[name]["train_seconds"] = time.time() - t0
        print(name, json.dumps({k: v for k, v in results[name].items()}, default=str))

    if not only or "dnn_only" in only:
        from src.models import build_dnn
        set_seed()
        model = build_dnn(tr["tab"].shape[1])
        hist = fit_keras(model, tr["tab"], tr["y"], va["tab"], va["y"], model_dir / "dnn_only.keras",
                         class_weights(tr["y"]))
        s_val = model.predict(va["tab"], batch_size=8192, verbose=0).ravel()
        np.save(model_dir / "dnn_only_val_scores.npy", s_val)
        results["dnn_only"] = evaluate(va["y"], s_val, best_f1_threshold(va["y"], s_val))
        results["dnn_only"]["train_seconds"] = hist["train_seconds"]
        print("dnn_only", json.dumps(results["dnn_only"], default=str))

    results_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    main(args.dev, args.only)
