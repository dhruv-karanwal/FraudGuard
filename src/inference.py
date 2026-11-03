"""Single-transaction scoring for the app: raw fields in, probability + attention out."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.features import FeatureScaler, raw_features
from src.sequences import gather_sequences

RAW_FIELDS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
              "oldbalanceDest", "newbalanceDest"]


def available_tag() -> str | None:
    for tag in ("full", "dev"):
        if (config.MODELS_DIR / tag / "hybrid_gru.keras").exists():
            return tag
    return None


@dataclass
class Prediction:
    probability: float
    threshold: float
    is_fraud: bool
    time_attention: list[float]   # over the window, oldest -> current
    fusion_weights: dict[str, float]
    window_len: int
    scored: bool                  # False for types the model does not score


class Predictor:
    def __init__(self, tag: str, model_name: str = "hybrid_gru"):
        import keras
        import src.models as m

        d = config.MODELS_DIR / tag
        self.tag = tag
        self.meta = json.loads((d / "prep_meta.json").read_text())
        self.seq_len = int(self.meta["seq_len"])
        self.tab_scaler = FeatureScaler.load(d / "tab_scaler.joblib")
        self.seq_scaler = FeatureScaler.load(d / "seq_scaler.joblib")
        self.model = keras.models.load_model(d / f"{model_name}.keras")
        self.explain = m.explainer(self.model)
        self.threshold = self._threshold(tag, model_name, d)

    @staticmethod
    def _threshold(tag: str, model_name: str, model_dir: Path) -> float:
        for p in (config.REPORTS_DIR / f"{tag}_test_metrics.json", config.REPORTS_DIR / f"{tag}_val_metrics.json"):
            if p.exists():
                res = json.loads(p.read_text())
                if model_name in res:
                    return float(res[model_name]["threshold"])
        return 0.5

    def arrays(self, current: dict, history: list[dict] | None = None):
        """Build model inputs for one transaction and its prior transactions
        (oldest first; at most ``seq_len - 1`` are used)."""
        history = list(history or [])[-(self.seq_len - 1):] if self.seq_len > 1 else []
        rows = pd.DataFrame([*history, current])[RAW_FIELDS]
        rows["type"] = rows["type"].astype(str)
        feats = raw_features(rows)
        tab = self.tab_scaler.transform(feats.iloc[[-1]])
        seq_feats = self.seq_scaler.transform(feats)
        n = len(rows)
        idx = np.full((1, self.seq_len), -1, dtype=np.int64)
        idx[0, self.seq_len - n:] = np.arange(n)
        seq, mask = gather_sequences(seq_feats, rows["step"].to_numpy(), idx)
        return {"tab": tab, "seq": seq, "mask": mask}

    def predict(self, current: dict, history: list[dict] | None = None) -> Prediction:
        x = self.arrays(current, history)
        n = int(x["mask"].sum())
        if str(current["type"]) not in config.FRAUD_TYPES:
            return Prediction(0.0, self.threshold, False, [0.0] * self.seq_len,
                              {"transaction": 0.0, "history": 0.0}, n, scored=False)
        p, t_attn, f_attn = self.explain.predict(x, verbose=0)
        prob = float(np.ravel(p)[0])
        return Prediction(prob, self.threshold, prob >= self.threshold,
                          [float(v) for v in t_attn[0]],
                          {"transaction": float(f_attn[0, 0]), "history": float(f_attn[0, 1])},
                          n, scored=True)

    def predict_batch(self, x: dict, batch_size: int = 8192) -> np.ndarray:
        return self.model.predict(x, batch_size=batch_size, verbose=0).ravel()
