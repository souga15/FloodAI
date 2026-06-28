"""Decision threshold selection — maximizes F1 on the validation set only.
Never touches test data, so the threshold is chosen before test evaluation
ever happens, preventing threshold-leakage from test into the reported metric."""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger("floodai.training.threshold")


def select_f1_optimal_threshold(y_val, y_val_pred_proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_pred_proba)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = f1s[:-1].argmax()  # thresholds has len(precision)-1 entries
    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Selected decision threshold tau*=%.4f (val F1=%.4f) — selected on "
        "VALIDATION set only, before any test-set evaluation.",
        best_threshold, f1s[best_idx],
    )
    return best_threshold
