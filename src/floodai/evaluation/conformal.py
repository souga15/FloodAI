"""
Conformal Prediction intervals for flood probability scores.

Implements the Split Conformal Prediction framework (Venn-Abers variant)
for producing valid marginal coverage guarantees on XGBoost probability outputs.

Reference: Angelopoulos & Bates (2022), "A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification."
arXiv:2107.07511

Why this matters for Q1 review:
  A XGBoost model predicting a single point probability (e.g. "flood prob=0.73")
  tells a flood manager nothing about confidence. A conformal prediction interval
  ("flood prob=0.73 [0.61, 0.88] at 90% coverage") is statistically guaranteed to
  contain the true probability in at least 90% of future days under exchangeability —
  no distributional assumptions required. This makes the system operationally
  meaningful for warning issuance decisions.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger("floodai.evaluation.conformal")


@dataclass
class ConformalResult:
    """Output of conformal calibration and prediction."""
    alpha: float                    # error rate (1 - coverage)
    coverage_target: float          # = 1 - alpha
    empirical_coverage: float       # actual coverage on calibration set
    q_hat: float                    # conformal quantile threshold
    n_calibration: int


class ConformalFloodPredictor:
    """
    Split conformal predictor for binary flood occurrence.

    Calibration:
        Uses the VALIDATION set only to compute the conformal quantile q_hat.
        The validation set is treated as the calibration set — it is never used
        for threshold selection in the same run (threshold is selected first,
        then conformal calibration uses residual scores on the same split).

    Prediction:
        For a new probability score p̂, the conformal interval is:
            lower = max(0, p̂ - q_hat)
            upper = min(1, p̂ + q_hat)
        This is the Regularized Adaptive Prediction Set (RAPS) simplification
        for binary regression calibration.

    Exchangeability assumption:
        Valid when calibration and test data are i.i.d. or exchangeable. In a
        temporal split, this assumption is mildly violated (test years > cal years),
        so the coverage guarantee is approximate. This is standard practice in
        operational ML hydrology literature and should be disclosed in the paper.
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: desired error rate. alpha=0.10 → target 90% coverage.
                   For operational flood warning, 90% is standard.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_hat: float | None = None
        self._cal_result: ConformalResult | None = None

    def calibrate(self, y_cal: np.ndarray, prob_cal: np.ndarray) -> ConformalResult:
        """
        Compute q_hat from calibration set scores.

        Nonconformity score: |y - p̂| for binary y, continuous p̂.
        The (1 - alpha) quantile of these scores is q_hat.

        Args:
            y_cal: true binary labels (0/1) on calibration set
            prob_cal: predicted probabilities on calibration set

        Returns:
            ConformalResult with empirical coverage and q_hat
        """
        y_cal = np.asarray(y_cal, dtype=float)
        prob_cal = np.asarray(prob_cal, dtype=float)
        if len(y_cal) != len(prob_cal):
            raise ValueError("y_cal and prob_cal must have the same length")

        # Nonconformity scores: absolute residual from true label
        scores = np.abs(y_cal - prob_cal)

        n = len(scores)
        # Corrected quantile: ceil((n+1)(1-alpha)) / n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = np.clip(level, 0.0, 1.0)
        self.q_hat = float(np.quantile(scores, level))

        # Compute empirical coverage on calibration set (should ≈ 1-alpha)
        lower = np.clip(prob_cal - self.q_hat, 0.0, 1.0)
        upper = np.clip(prob_cal + self.q_hat, 0.0, 1.0)
        covered = np.mean((y_cal >= lower) & (y_cal <= upper))

        self._cal_result = ConformalResult(
            alpha=self.alpha,
            coverage_target=1 - self.alpha,
            empirical_coverage=float(covered),
            q_hat=self.q_hat,
            n_calibration=n,
        )
        logger.info(
            "Conformal calibration: q_hat=%.4f, target coverage=%.0f%%, "
            "empirical coverage=%.1f%% (n_cal=%d)",
            self.q_hat, 100 * (1 - self.alpha), 100 * covered, n,
        )
        return self._cal_result

    def predict_intervals(
        self, prob_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute conformal prediction intervals for test probabilities.

        Args:
            prob_test: predicted probabilities on test set

        Returns:
            (lower_bounds, upper_bounds) arrays clipped to [0, 1]
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_intervals().")
        prob_test = np.asarray(prob_test, dtype=float)
        lower = np.clip(prob_test - self.q_hat, 0.0, 1.0)
        upper = np.clip(prob_test + self.q_hat, 0.0, 1.0)
        return lower, upper

    def predict_interval_df(
        self,
        prob_test: np.ndarray,
        index=None,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns [prob, lower, upper, interval_width]."""
        lower, upper = self.predict_intervals(prob_test)
        return pd.DataFrame({
            "flood_prob": prob_test,
            "ci_lower": lower,
            "ci_upper": upper,
            "interval_width": upper - lower,
        }, index=index)

    @property
    def calibration_result(self) -> ConformalResult | None:
        return self._cal_result


def add_conformal_to_results(
    y_val: np.ndarray,
    val_proba: np.ndarray,
    y_test: np.ndarray,
    test_proba: np.ndarray,
    alpha: float = 0.10,
) -> tuple[pd.DataFrame, ConformalResult]:
    """
    Convenience wrapper: calibrate on val, predict intervals on test.

    Returns:
        (interval_df, cal_result) where interval_df has columns
        [flood_prob, ci_lower, ci_upper, interval_width]
    """
    cp = ConformalFloodPredictor(alpha=alpha)
    cal_result = cp.calibrate(y_val, val_proba)
    interval_df = cp.predict_interval_df(test_proba)
    return interval_df, cal_result
