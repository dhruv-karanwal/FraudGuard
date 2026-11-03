"""Stage 4: train the hybrid DNN + GRU/LSTM attention model.

Run: python -m src.train [--dev] [--rnn gru|lstm]

Selects the threshold on validation and reports validation metrics only; the
test set is evaluated once in src.evaluate.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src import config
from src.metrics import best_f1_threshold, class_weights, evaluate
from src.prepare import load_split
from src.training import fit_keras, set_seed


def inputs(d: dict) -> dict:
    return {"tab": d["tab"], "seq": d["seq"], "mask": d["mask"]}


def main(dev: bool, rnn: str = "gru", epochs: int = 30) -> None:
    from src.models import build_hybrid

    set_seed()
    tag = "dev" if dev else "full"
    model_dir = config.MODELS_DIR / tag
    tr, va = load_split(tag, "train"), load_split(tag, "val")
    _, seq_len, n_seq = tr["seq"].shape

    model = build_hybrid(tr["tab"].shape[1], seq_len, n_seq, rnn=rnn)
    model.summary()
    name = f"hybrid_{rnn}"
    hist = fit_keras(model, inputs(tr), tr["y"], inputs(va), va["y"], model_dir / f"{name}.keras",
                     class_weights(tr["y"]), epochs=epochs)

    s_val = model.predict(inputs(va), batch_size=8192, verbose=0).ravel()
    np.save(model_dir / f"{name}_val_scores.npy", s_val)
    res = evaluate(va["y"], s_val, best_f1_threshold(va["y"], s_val))
    res["train_seconds"] = hist["train_seconds"]
    res["epochs_run"] = len(hist["loss"])
    print(name, json.dumps(res))

    path = config.REPORTS_DIR / f"{tag}_val_metrics.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results[name] = res
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--rnn", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    main(args.dev, args.rnn, args.epochs)
