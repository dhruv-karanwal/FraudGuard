"""Per-account history windows for the recurrent branch.

For every row, the window holds the last ``seq_len`` transactions of the same
account (current row last, oldest first), left-padded. Windows never cross a
split boundary and never contain a later transaction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def history_index(df: pd.DataFrame, key: str = "nameOrig", seq_len: int = 5,
                  split_col: str = "split") -> np.ndarray:
    """Return an ``(n, seq_len)`` int64 array of row positions (``-1`` = pad).

    ``df`` must already be in chronological order; position in ``df`` is the
    time order. Column ``seq_len - 1`` is always the row itself.
    """
    n = len(df)
    pos_time = np.arange(n, dtype=np.int64)
    key_codes = pd.factorize(df[key])[0].astype(np.int64)
    if split_col in df.columns:
        split_codes = pd.factorize(df[split_col])[0].astype(np.int64)
    else:
        split_codes = np.zeros(n, dtype=np.int64)

    # Group by (split, key), keep time order inside each group.
    order = np.lexsort((pos_time, key_codes, split_codes))
    s, k = split_codes[order], key_codes[order]
    new_group = np.ones(n, dtype=bool)
    new_group[1:] = (s[1:] != s[:-1]) | (k[1:] != k[:-1])
    ar = np.arange(n, dtype=np.int64)
    group_start = np.maximum.accumulate(np.where(new_group, ar, 0))
    pos_in_group = ar - group_start

    idx_sorted = np.full((n, seq_len), -1, dtype=np.int64)
    for lag in range(seq_len):
        valid = pos_in_group >= lag
        src = np.where(valid, ar - lag, 0)
        idx_sorted[:, seq_len - 1 - lag] = np.where(valid, order[src], -1)

    idx = np.empty_like(idx_sorted)
    idx[order] = idx_sorted
    return idx


def gather_sequences(features: np.ndarray, steps: np.ndarray, idx: np.ndarray,
                     rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(seq, mask)`` for the selected ``rows``.

    ``seq`` has shape ``(m, seq_len, f + 1)``; the extra last channel is
    ``log1p(hours since that history item)`` (0 for the current row). Padded
    positions are all zeros and ``mask`` is 0 there.
    """
    if rows is not None:
        idx = idx[rows]
    m, L = idx.shape
    f = features.shape[1]
    valid = idx >= 0
    safe = np.where(valid, idx, 0)
    seq = np.zeros((m, L, f + 1), dtype=np.float32)
    seq[:, :, :f] = features[safe]
    cur_step = steps[idx[:, -1]].astype(np.float32)
    seq[:, :, f] = np.log1p(np.maximum(cur_step[:, None] - steps[safe].astype(np.float32), 0))
    seq[~valid] = 0.0
    return seq, valid.astype(np.float32)
