"""Stage 2: build model-ready arrays for each split.

Run: python -m src.prepare [--dev]

Writes to data/processed/<tag>/:
  {split}_tab.npy   (m, n_tab)        scaled transaction features
  {split}_seq.npy   (m, L, n_seq+1)   scaled history windows (+ time-delta channel)
  {split}_mask.npy  (m, L)            1 = real history item
  {split}_y.npy     (m,)              labels
  {split}_meta.parquet                raw columns of the scored rows (for the app)
and to models/<tag>/: tab_scaler.joblib, seq_scaler.joblib, prep_meta.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src import config
from src.data import SPLITS, chronological_split, dev_sample, load_paysim
from src.features import SEQ_FEATURES, TAB_FEATURES, FeatureScaler, raw_features
from src.sequences import gather_sequences, history_index

META_COLS = ["row_id", "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
             "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]


def paths(tag: str):
    data_dir = config.PROCESSED_DIR / tag
    model_dir = config.MODELS_DIR / tag
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, model_dir


def load_split(tag: str, split: str, mmap: bool = False) -> dict:
    data_dir = config.PROCESSED_DIR / tag
    mode = "r" if mmap else None
    out = {k: np.load(data_dir / f"{split}_{k}.npy", mmap_mode=mode) for k in ("tab", "seq", "mask", "y")}
    return out


def main(dev: bool, seq_len: int = config.SEQ_LEN, key: str = "nameOrig") -> None:
    t0 = time.time()
    tag = "dev" if dev else "full"
    data_dir, model_dir = paths(tag)

    df = load_paysim()
    if dev:
        df = dev_sample(df)
    df, bounds = chronological_split(df)
    print(f"[{time.time() - t0:6.1f}s] loaded {len(df):,} rows; boundaries {bounds}")

    feats = raw_features(df)
    scored = df["type"].isin(config.FRAUD_TYPES).to_numpy()
    is_train = (df["split"] == "train").to_numpy()

    # Fit on train only: tab scaler on scored train rows, seq scaler on all train rows.
    tab_scaler = FeatureScaler(TAB_FEATURES).fit(feats[scored & is_train])
    seq_scaler = FeatureScaler(SEQ_FEATURES).fit(feats[is_train])
    tab_scaler.save(model_dir / "tab_scaler.joblib")
    seq_scaler.save(model_dir / "seq_scaler.joblib")

    seq_all = seq_scaler.transform(feats)
    idx = history_index(df, key=key, seq_len=seq_len)
    steps = df["step"].to_numpy()
    print(f"[{time.time() - t0:6.1f}s] features + history index built")

    counts = {}
    for split in SPLITS:
        rows = np.flatnonzero(scored & (df["split"] == split).to_numpy())
        tab = tab_scaler.transform(feats.iloc[rows])
        seq, mask = gather_sequences(seq_all, steps, idx, rows)
        y = df[config.TARGET].to_numpy()[rows].astype(np.float32)
        np.save(data_dir / f"{split}_tab.npy", tab)
        np.save(data_dir / f"{split}_seq.npy", seq)
        np.save(data_dir / f"{split}_mask.npy", mask)
        np.save(data_dir / f"{split}_y.npy", y)
        df.iloc[rows][META_COLS].reset_index(drop=True).to_parquet(data_dir / f"{split}_meta.parquet")
        hist_len = mask.sum(1)
        counts[split] = {
            "rows": int(len(rows)), "fraud": int(y.sum()), "fraud_rate": float(y.mean()),
            "steps": [int(steps[rows].min()), int(steps[rows].max())],
            "with_history": int((hist_len > 1).sum()),
            "mean_window_len": float(hist_len.mean()),
        }
        print(f"[{time.time() - t0:6.1f}s] {split}: {counts[split]}")

    meta = {
        "tag": tag, "seq_len": seq_len, "group_key": key,
        "split_boundaries": {"last_train_step": bounds[0], "last_val_step": bounds[1]},
        "tab_features": TAB_FEATURES, "seq_features": SEQ_FEATURES + ["log_hours_since"],
        "splits": counts,
    }
    (model_dir / "prep_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="use the development sample")
    ap.add_argument("--seq-len", type=int, default=config.SEQ_LEN)
    args = ap.parse_args()
    main(args.dev, args.seq_len)
