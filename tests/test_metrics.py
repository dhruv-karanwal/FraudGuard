import numpy as np
import pytest

from src.metrics import best_f1_threshold, class_weights, evaluate


def test_confusion_matrix_layout():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    m = evaluate(y, s, 0.5)
    # [[TN, FP], [FN, TP]]
    assert m["confusion_matrix"] == [[2, 1], [1, 1]]
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)


def test_best_threshold_separates_perfectly():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    t = best_f1_threshold(y, s)
    assert evaluate(y, s, t)["f1"] == 1.0


def test_class_weights_balance():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y)
    assert w[0] * 90 == pytest.approx(w[1] * 10)
