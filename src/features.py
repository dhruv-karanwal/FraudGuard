"""Per-transaction feature engineering and train-only scaling."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config


def _slog(x: np.ndarray) -> np.ndarray:
    """Signed log1p: keeps sign, compresses magnitude."""
    return np.sign(x) * np.log1p(np.abs(x))


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-local features (no fitted state, no label, no
    isFlaggedFraud). Safe to compute before splitting."""
    amount = df["amount"].to_numpy(dtype=np.float64)
    ob_o = df["oldbalanceOrg"].to_numpy(dtype=np.float64)
    nb_o = df["newbalanceOrig"].to_numpy(dtype=np.float64)
    ob_d = df["oldbalanceDest"].to_numpy(dtype=np.float64)
    nb_d = df["newbalanceDest"].to_numpy(dtype=np.float64)
    hour = (df["step"].to_numpy() % 24).astype(np.float64)
    types = df["type"].astype(str).to_numpy()

    out = {
        "log_amount": np.log1p(amount),
        "log_oldbalanceOrg": np.log1p(ob_o),
        "log_newbalanceOrig": np.log1p(nb_o),
        "log_oldbalanceDest": np.log1p(ob_d),
        "log_newbalanceDest": np.log1p(nb_d),
        # Accounting mismatches; large for most simulated fraud.
        "err_balance_orig": _slog(nb_o + amount - ob_o),
        "err_balance_dest": _slog(ob_d + amount - nb_d),
        "amount_to_balance": np.log1p(amount / (ob_o + 1.0)),
        "orig_zero_before": (ob_o == 0).astype(np.float64),
        "orig_emptied": ((nb_o == 0) & (ob_o > 0)).astype(np.float64),
        "dest_zero_both": ((ob_d == 0) & (nb_d == 0)).astype(np.float64),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }
    for t in config.ALL_TYPES:
        out[f"type_{t}"] = (types == t).astype(np.float64)
    return pd.DataFrame(out, index=df.index)


# Tabular branch: scored rows are only TRANSFER / CASH_OUT, so one type flag is enough.
TAB_FEATURES = [
    "log_amount", "log_oldbalanceOrg", "log_newbalanceOrig", "log_oldbalanceDest",
    "log_newbalanceDest", "err_balance_orig", "err_balance_dest", "amount_to_balance",
    "orig_zero_before", "orig_emptied", "dest_zero_both", "hour_sin", "hour_cos",
    "type_TRANSFER",
]

# Sequence branch: history may contain any transaction type.
SEQ_FEATURES = [
    "log_amount", "log_oldbalanceOrg", "log_newbalanceOrig", "err_balance_orig",
    "amount_to_balance", "hour_sin", "hour_cos",
] + [f"type_{t}" for t in config.ALL_TYPES]

# Binary / bounded columns are left unscaled.
_UNSCALED = {"orig_zero_before", "orig_emptied", "dest_zero_both", "hour_sin", "hour_cos"} | {
    f"type_{t}" for t in config.ALL_TYPES
}


class FeatureScaler:
    """StandardScaler over the continuous columns, fit on train rows only."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.scaled = [c for c in self.columns if c not in _UNSCALED]
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, feats: pd.DataFrame) -> "FeatureScaler":
        self.scaler.fit(feats[self.scaled].to_numpy())
        self.fitted = True
        return self

    def transform(self, feats: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureScaler.transform called before fit")
        out = feats[self.columns].to_numpy(dtype=np.float64).copy()
        idx = [self.columns.index(c) for c in self.scaled]
        out[:, idx] = self.scaler.transform(feats[self.scaled].to_numpy())
        return out.astype(np.float32)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FeatureScaler":
        return joblib.load(path)
