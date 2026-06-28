"""
Evaluation metrics with enforced data provenance.

This module exists specifically because of what was found in
WPT-D-26-00166 Cell 23: a bootstrap computed over the full dataset
(including training rows), producing AUC=0.9815/F1=0.7018, sitting
alongside the legitimate held-out test result (AUC=0.9623/F1=0.4878),
with the inflated number apparently the one written into the manuscript.

The mechanism here is not a comment or a naming convention — it is a type.
`EvaluationResult.provenance` is a required field. `report_headline()` 
raises if provenance != "held_out". There is no code path in this module
that lets a training-inclusive result reach the "headline" label.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

logger = logging.getLogger("floodai.evaluation.metrics")


class DataProvenance(str, Enum):
    HELD_OUT = "held_out"             # validation or test set, never seen by the fitted model
    TRAINING_INCLUSIVE = "training_inclusive"  # includes rows the model was fit on — diagnostic only
    LOSO_HELD_OUT = "loso_held_out"   # left-out basin in leave-one-basin-out CV


class HeadlineMetricError(Exception):
    """Raised when code attempts to report a non-held-out result as a headline metric."""


@dataclass(frozen=True)
class EvaluationResult:
    set_name: str
    provenance: DataProvenance
    n_samples: int
    n_positive: int
    roc_auc: float
    pr_auc: float
    f1: float
    mcc: float
    balanced_accuracy: float
    far: float  # false alarm rate
    csi: float  # critical success index
    threshold: float
    confusion: tuple[int, int, int, int]  # tn, fp, fn, tp

    def __post_init__(self) -> None:
        if self.provenance == DataProvenance.TRAINING_INCLUSIVE:
            logger.warning(
                "EvaluationResult for '%s' has provenance=TRAINING_INCLUSIVE. "
                "This result must NEVER be reported as a headline metric. "
                "It may only be used as an internal overfitting diagnostic "
                "(e.g. comparing train vs held-out recall gap).",
                self.set_name,
            )


def evaluate(
    y_true,
    y_pred_proba,
    threshold: float,
    set_name: str,
    provenance: DataProvenance,
) -> EvaluationResult:
    """Compute the full metric suite for one set. Always tags the result with
    its provenance — there is no "untagged" evaluation path in this codebase."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    if y_true.sum() == 0:
        raise ValueError(f"Set '{set_name}' has zero positive samples; metrics like AUC are undefined.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tp + 1e-9)
    csi = tp / (tp + fp + fn + 1e-9)

    result = EvaluationResult(
        set_name=set_name,
        provenance=provenance,
        n_samples=len(y_true),
        n_positive=int(y_true.sum()),
        roc_auc=roc_auc_score(y_true, y_pred_proba),
        pr_auc=average_precision_score(y_true, y_pred_proba),
        f1=f1_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        far=far,
        csi=csi,
        threshold=threshold,
        confusion=(int(tn), int(fp), int(fn), int(tp)),
    )
    logger.info(
        "Evaluated '%s' [%s]: AUC=%.4f F1=%.4f MCC=%.4f n=%d (pos=%d)",
        set_name, provenance.value, result.roc_auc, result.f1, result.mcc,
        result.n_samples, result.n_positive,
    )
    return result


def report_headline(result: EvaluationResult) -> EvaluationResult:
    """
    The ONLY function permitted to label a result as a manuscript headline
    number. Raises HeadlineMetricError if the result's provenance is not
    HELD_OUT. Call this explicitly at the point in your reporting script
    where you write the number that will go in the paper's abstract/results
    table — if this call raises, that is the system telling you that you are
    about to report an invalid number, exactly as happened in Cell 23 of the
    prior notebook.
    """
    if result.provenance != DataProvenance.HELD_OUT:
        raise HeadlineMetricError(
            f"Refusing to report '{result.set_name}' (provenance={result.provenance.value}) "
            f"as a headline metric. Only provenance=HELD_OUT results may be headlined. "
            f"This guard exists specifically because of the WPT-D-26-00166 Cell 23 "
            f"issue (a training-inclusive bootstrap reported as AUC=0.9815/F1=0.7018 "
            f"alongside a legitimate test result of 0.9623/0.4878). If you believe this "
            f"result should be reportable, the fix is to change how it was computed, "
            f"not to bypass this check."
        )
    logger.info("HEADLINE METRIC APPROVED: '%s' AUC=%.4f F1=%.4f (provenance=held_out)",
                result.set_name, result.roc_auc, result.f1)
    return result


def bootstrap_ci(
    y_true,
    y_pred_proba,
    threshold: float,
    n_resamples: int,
    seed: int,
    provenance: DataProvenance,
) -> dict[str, tuple[float, float, float]]:
    """
    Bootstrap confidence intervals — resamples ONLY from the arrays passed in.
    The caller is responsible for ensuring y_true/y_pred_proba come from a
    held-out set; this function additionally requires an explicit
    `provenance` argument and refuses to proceed if it is TRAINING_INCLUSIVE,
    so this utility cannot be reused to recreate the Cell 23 pattern under a
    different function name.
    """
    if provenance == DataProvenance.TRAINING_INCLUSIVE:
        raise HeadlineMetricError(
            "bootstrap_ci() called with provenance=TRAINING_INCLUSIVE. "
            "Bootstrapping over training-inclusive data measures memorization, "
            "not generalization, and produces inflated, non-reportable "
            "confidence intervals. Pass held-out data only."
        )

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    n = len(y_true)

    aucs, f1s, praucs = [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        pred = (yp >= threshold).astype(int)
        aucs.append(roc_auc_score(yt, yp))
        f1s.append(f1_score(yt, pred, zero_division=0))
        praucs.append(average_precision_score(yt, yp))

    def _summary(vals: list[float]) -> tuple[float, float, float]:
        return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    out = {"roc_auc": _summary(aucs), "f1": _summary(f1s), "pr_auc": _summary(praucs)}
    logger.info(
        "Bootstrap CI (provenance=%s, n=%d): AUC=%.4f [%.4f, %.4f], F1=%.4f [%.4f, %.4f]",
        provenance.value, n_resamples, *out["roc_auc"], *out["f1"],
    )
    return out
