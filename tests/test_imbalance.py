"""Additional tests for floodai.training.imbalance — guarding against the
real incident where an un-parameterized SMOTE() call (bypassing this module
entirely) silently defaulted to 50/50 balancing instead of the configured
~10% ratio, contributing to a spurious LOBO AUC=1.000 result."""
from __future__ import annotations

import numpy as np
import pytest

from floodai.training.imbalance import resample_training_only


def _imbalanced_data(n=2000, pos_rate=0.05, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    X = rng.normal(0, 1, size=(n, n_features))
    return X, y


class TestSmoteRatioGuard:
    def test_correct_ratio_passes(self):
        X, y = _imbalanced_data(pos_rate=0.05)
        X_res, y_res = resample_training_only(X, y, sampling_strategy=0.10, k_neighbors_max=5, seed=42)
        actual_ratio = y_res.sum() / (len(y_res) - y_res.sum())
        assert abs(actual_ratio - 0.10) < 0.02

    def test_skips_smote_with_too_few_positives(self):
        X, y = _imbalanced_data(n=500, pos_rate=0.005)  # ~2-3 positives
        X_res, y_res = resample_training_only(X, y, sampling_strategy=0.10, k_neighbors_max=5, seed=42)
        np.testing.assert_array_equal(y_res, y)  # unchanged — SMOTE skipped

    def test_reproducible_given_same_seed(self):
        X, y = _imbalanced_data(pos_rate=0.05)
        X_res1, y_res1 = resample_training_only(X, y, sampling_strategy=0.10, k_neighbors_max=5, seed=7)
        X_res2, y_res2 = resample_training_only(X, y, sampling_strategy=0.10, k_neighbors_max=5, seed=7)
        np.testing.assert_array_equal(y_res1, y_res2)
        np.testing.assert_allclose(X_res1, X_res2)
