"""
Class imbalance handling — SMOTE applied to the training partition only.

Enforcement mechanism: `resample_training_only()` takes ONLY X_train/y_train
as arguments. There is no function signature in this module that accepts
validation or test data alongside a resampling call, so it is structurally
impossible to accidentally pass held-out data through SMOTE from this module.
"""
from __future__ import annotations

import logging

import numpy as np
from imblearn.over_sampling import SMOTE

logger = logging.getLogger("floodai.training.imbalance")


def resample_training_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float,
    k_neighbors_max: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to (X_train, y_train) ONLY. Signature deliberately excludes
    any validation/test arguments."""
    n_positive = int(np.sum(y_train == 1))
    if n_positive < 6:
        logger.warning(
            "Only %d positive samples in training data; skipping SMOTE "
            "(need at least k_neighbors+1). Returning original training data unchanged.",
            n_positive,
        )
        return X_train, y_train

    k = min(k_neighbors_max, n_positive - 1)
    smote = SMOTE(random_state=seed, k_neighbors=k, sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(
        "SMOTE applied to TRAINING DATA ONLY: %d -> %d positive samples "
        "(sampling_strategy=%.2f, k=%d). Validation/test sets are untouched "
        "by construction (this function does not accept them as arguments).",
        n_positive, int(np.sum(y_res == 1)), sampling_strategy, k,
    )
    return X_res, y_res
