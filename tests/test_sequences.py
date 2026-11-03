import numpy as np
import pandas as pd

from src.sequences import gather_sequences, history_index


def _df():
    # Chronological order is row order. Account A spans train and val.
    return pd.DataFrame({
        "step":     [1, 1, 2, 3, 4, 5, 6, 7],
        "nameOrig": ["A", "B", "A", "A", "B", "A", "A", "C"],
        "split":    ["train", "train", "train", "train", "train", "val", "val", "test"],
    })


def test_current_row_is_last_and_history_is_past_only():
    df = _df()
    idx = history_index(df, seq_len=3)
    assert (idx[:, -1] == np.arange(len(df))).all()
    for i, row in enumerate(idx):
        real = row[row >= 0]
        assert (np.diff(real) > 0).all(), "history must be strictly increasing in time"
        assert (real <= i).all(), "no future rows"
        assert (df.nameOrig.iloc[real] == df.nameOrig.iloc[i]).all()


def test_expected_windows():
    idx = history_index(_df(), seq_len=3)
    assert idx[0].tolist() == [-1, -1, 0]
    assert idx[2].tolist() == [-1, 0, 2]
    assert idx[3].tolist() == [0, 2, 3]
    assert idx[4].tolist() == [-1, 1, 4]
    # First val row of A must not see A's train history.
    assert idx[5].tolist() == [-1, -1, 5]
    assert idx[6].tolist() == [-1, 5, 6]


def test_no_window_crosses_split_boundary():
    df = _df()
    idx = history_index(df, seq_len=4)
    split = df.split.to_numpy()
    for i, row in enumerate(idx):
        real = row[row >= 0]
        assert (split[real] == split[i]).all()


def test_gather_pads_with_zero_and_masks():
    df = _df()
    idx = history_index(df, seq_len=3)
    feats = np.arange(len(df) * 2, dtype=np.float32).reshape(-1, 2) + 1.0
    seq, mask = gather_sequences(feats, df.step.to_numpy(), idx, rows=np.array([0, 3]))
    assert seq.shape == (2, 3, 3)
    assert mask.tolist() == [[0, 0, 1], [1, 1, 1]]
    assert (seq[0, :2] == 0).all()
    np.testing.assert_allclose(seq[1, :, :2], feats[[0, 2, 3]])
    # Time-delta channel: current row 0, older rows log1p(hours ago).
    np.testing.assert_allclose(seq[1, :, 2], np.log1p([2, 1, 0]), rtol=1e-6)


def test_without_split_column_uses_single_group():
    df = _df().drop(columns="split")
    idx = history_index(df, seq_len=3)
    assert idx[5].tolist() == [2, 3, 5]
