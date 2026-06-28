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

    n_negative = int(np.sum(y_train == 0))
    natural_ratio = float(n_positive) / float(n_negative) if n_negative > 0 else 1.0
    if natural_ratio >= sampling_strategy:
        logger.warning(
            "Natural positive ratio (%.3f) is already >= sampling_strategy (%.3f). "
            "Skipping SMOTE to prevent imblearn downsampling errors.",
            natural_ratio, sampling_strategy
        )
        return X_train, y_train

    k = min(k_neighbors_max, n_positive - 1)
    smote = SMOTE(random_state=seed, k_neighbors=k, sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    actual_ratio = float(np.sum(y_res == 1)) / float(np.sum(y_res == 0))
    # If the caller passed e.g. sampling_strategy=0.10 but the resulting ratio
    # is way off (most commonly: someone bypassed this function and called
    # SMOTE() with no sampling_strategy elsewhere, defaulting to 1.0 / 50-50),
    # catch it here too as a second line of defense.
    if abs(actual_ratio - sampling_strategy) > 0.05:
        raise RuntimeError(
            f"SMOTE produced a positive:negative ratio of {actual_ratio:.3f}, "
            f"which does not match the requested sampling_strategy="
            f"{sampling_strategy:.3f}. This mismatch is exactly the failure "
            f"mode that previously caused spurious LOBO AUC=1.000 results "
            f"(an un-parameterized `SMOTE(random_state=...)` call elsewhere "
            f"in a notebook silently defaulted to 50/50 balancing instead of "
            f"the intended ~10%). Use this function for ALL SMOTE calls in "
            f"this project -- do not call imblearn.SMOTE directly."
        )

    logger.info(
        "SMOTE applied to TRAINING DATA ONLY: %d -> %d positive samples "
        "(sampling_strategy=%.2f, k=%d, verified actual_ratio=%.3f). "
        "Validation/test sets are untouched by construction (this function "
        "does not accept them as arguments).",
        n_positive, int(np.sum(y_res == 1)), sampling_strategy, k, actual_ratio,
    )
    return X_res, y_res
