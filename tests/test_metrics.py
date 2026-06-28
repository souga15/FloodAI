"""
Tests for floodai.evaluation.metrics — the core regression-prevention tests
for this entire project. These tests must pass for the framework's central
claim (no train-inclusive number can be reported as a headline result) to
be true rather than aspirational.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodai.evaluation.metrics import (
    DataProvenance,
    HeadlineMetricError,
    bootstrap_ci,
    evaluate,
    report_headline,
)


def _synthetic_predictions(n=500, pos_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < pos_rate).astype(int)
    # Make probabilities weakly informative so AUC isn't degenerate.
    proba = np.clip(y_true * 0.6 + rng.normal(0, 0.2, n) + 0.1, 0, 1)
    return y_true, proba


class TestHeadlineGuard:
    def test_held_out_result_can_be_headlined(self):
        y_true, proba = _synthetic_predictions()
        result = evaluate(y_true, proba, threshold=0.3, set_name="test_set", provenance=DataProvenance.HELD_OUT)
        headlined = report_headline(result)
        assert headlined.set_name == "test_set"

    def test_training_inclusive_result_cannot_be_headlined(self):
        """This is the direct regression test for the Cell 23 issue."""
        y_true, proba = _synthetic_predictions()
        result = evaluate(
            y_true, proba, threshold=0.3, set_name="full_dataset_bootstrap",
            provenance=DataProvenance.TRAINING_INCLUSIVE,
        )
        with pytest.raises(HeadlineMetricError, match="Cell 23"):
            report_headline(result)

    def test_loso_result_cannot_be_headlined_as_primary(self):
        """LOSO is a valid generalization check but is a different claim than
        the primary temporal-split headline; it must not be silently
        substituted in if the primary result looks worse."""
        y_true, proba = _synthetic_predictions()
        result = evaluate(
            y_true, proba, threshold=0.3, set_name="loso_fold",
            provenance=DataProvenance.LOSO_HELD_OUT,
        )
        with pytest.raises(HeadlineMetricError):
            report_headline(result)


class TestBootstrapGuard:
    def test_bootstrap_refuses_training_inclusive_provenance(self):
        y_true, proba = _synthetic_predictions(n=1000)
        with pytest.raises(HeadlineMetricError, match="memorization"):
            bootstrap_ci(
                y_true, proba, threshold=0.3, n_resamples=100, seed=42,
                provenance=DataProvenance.TRAINING_INCLUSIVE,
            )

    def test_bootstrap_succeeds_on_held_out_provenance(self):
        y_true, proba = _synthetic_predictions(n=1000)
        out = bootstrap_ci(
            y_true, proba, threshold=0.3, n_resamples=200, seed=42,
            provenance=DataProvenance.HELD_OUT,
        )
        mean_auc, lo, hi = out["roc_auc"]
        assert lo <= mean_auc <= hi

    def test_bootstrap_is_reproducible_given_same_seed(self):
        y_true, proba = _synthetic_predictions(n=500)
        out1 = bootstrap_ci(y_true, proba, 0.3, 100, seed=7, provenance=DataProvenance.HELD_OUT)
        out2 = bootstrap_ci(y_true, proba, 0.3, 100, seed=7, provenance=DataProvenance.HELD_OUT)
        assert out1["roc_auc"] == out2["roc_auc"], "Same seed must produce identical bootstrap CIs"


class TestEvaluateBasics:
    def test_raises_on_zero_positive_samples(self):
        y_true = np.zeros(100)
        proba = np.random.default_rng(0).random(100)
        with pytest.raises(ValueError, match="zero positive"):
            evaluate(y_true, proba, threshold=0.5, set_name="empty", provenance=DataProvenance.HELD_OUT)

    def test_confusion_counts_sum_to_n(self):
        y_true, proba = _synthetic_predictions(n=300)
        result = evaluate(y_true, proba, threshold=0.3, set_name="x", provenance=DataProvenance.HELD_OUT)
        assert sum(result.confusion) == 300
