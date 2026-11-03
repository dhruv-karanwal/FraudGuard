"""Loading, sampling and chronological splitting of PaySim."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

SPLITS = ("train", "val", "test")


def load_paysim(path: Path | str = config.RAW_CSV, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw CSV, cached as parquet next to the processed data.

    Adds ``row_id`` (original file order) so ties within a step keep the
    simulator's order.
    """
    path = Path(path)
    cache = config.PROCESSED_DIR / "paysim_full.parquet"
    if nrows is None and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache)
    df = pd.read_csv(path, dtype=DTYPES, nrows=nrows)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    if nrows is None:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def dev_sample(df: pd.DataFrame, legit_every: int = config.DEV_LEGIT_EVERY) -> pd.DataFrame:
    """All fraud rows plus every ``legit_every``-th legit row, in time order."""
    legit_idx = np.flatnonzero(df[config.TARGET].to_numpy() == 0)
    keep = np.zeros(len(df), dtype=bool)
    keep[legit_idx[::legit_every]] = True
    keep |= df[config.TARGET].to_numpy() == 1
    return df.loc[keep].reset_index(drop=True)


def split_boundaries(steps: np.ndarray, train_frac: float = config.TRAIN_FRAC,
                     val_frac: float = config.VAL_FRAC) -> tuple[int, int]:
    """Return (last_train_step, last_val_step) so that roughly ``train_frac`` /
    ``val_frac`` of rows fall in each part. Boundaries are whole steps."""
    s = np.sort(np.asarray(steps))
    n = len(s)
    t_end = int(s[min(n - 1, int(n * train_frac))])
    v_end = int(s[min(n - 1, int(n * (train_frac + val_frac)))])
    if v_end <= t_end:
        raise ValueError("Split boundaries collapsed; not enough distinct steps.")
    return t_end, v_end


def assign_split(df: pd.DataFrame, boundaries: tuple[int, int]) -> pd.Series:
    t_end, v_end = boundaries
    step = df["step"].to_numpy()
    out = np.where(step <= t_end, "train", np.where(step <= v_end, "val", "test"))
    return pd.Series(pd.Categorical(out, categories=SPLITS), index=df.index, name="split")


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, int]]:
    """Sort by (step, row_id) and add a ``split`` column.

    Boundaries are computed on the scored rows (TRANSFER / CASH_OUT) so the
    model-facing fractions match the config.
    """
    df = df.sort_values(["step", "row_id"], kind="stable").reset_index(drop=True)
    scored = df["type"].isin(config.FRAUD_TYPES).to_numpy()
    bounds = split_boundaries(df.loc[scored, "step"].to_numpy())
    df["split"] = assign_split(df, bounds)
    return df, bounds
