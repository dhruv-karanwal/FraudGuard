import numpy as np
import pandas as pd
import pytest

from src.data import assign_split, chronological_split, dev_sample, split_boundaries


def _toy(n_steps=20, per_step=10):
    rows = []
    rid = 0
    for step in range(1, n_steps + 1):
        for j in range(per_step):
            rows.append({
                "row_id": rid, "step": step,
                "type": "TRANSFER" if j % 2 else "CASH_OUT",
                "isFraud": int(j == 0 and step % 5 == 0),
            })
            rid += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=42)  # shuffled on purpose


def test_split_is_chronological_and_disjoint():
    df, (t_end, v_end) = chronological_split(_toy())
    assert t_end < v_end
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    assert tr.step.max() < va.step.min() <= va.step.max() < te.step.min()
    assert len(tr) + len(va) + len(te) == len(df)
    # Sorted by time after split.
    assert (np.diff(df.step.to_numpy()) >= 0).all()


def test_split_fractions_roughly_match():
    df, _ = chronological_split(_toy(n_steps=100))
    frac = df.split.value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.02)
    assert frac["val"] == pytest.approx(0.15, abs=0.02)


def test_boundaries_fall_on_whole_steps():
    steps = np.repeat(np.arange(1, 11), 3)
    t_end, v_end = split_boundaries(steps)
    s = assign_split(pd.DataFrame({"step": steps}), (t_end, v_end))
    for step in np.unique(steps):
        assert s[steps == step].nunique() == 1


def test_dev_sample_keeps_all_fraud_and_time_order():
    df = _toy().sort_values("row_id").reset_index(drop=True)
    sample = dev_sample(df, legit_every=4)
    assert sample.isFraud.sum() == df.isFraud.sum()
    assert (np.diff(sample.row_id.to_numpy()) > 0).all()
    n_legit = (df.isFraud == 0).sum()
    assert (sample.isFraud == 0).sum() == int(np.ceil(n_legit / 4))
