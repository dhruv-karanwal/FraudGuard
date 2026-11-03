"""Metrics and threshold selection shared by every model."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 on the given (validation) data."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        # Standard layout: [[TN, FP], [FN, TP]]
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced weights: n / (2 * n_c)."""
    y = np.asarray(y).astype(int)
    n, n_pos = len(y), int(y.sum())
    return {0: n / (2 * (n - n_pos)), 1: n / (2 * n_pos)}
