"""
Leave-One-Basin-Out (LOBO) spatial cross-validation.

With only 3 basins (Ganga/Bihar, Brahmaputra/Assam, Mahanadi/Odisha), this
is a coarser check than the original 10-state LOSO in WPT-D-26-00166 — that
limitation is real and must be stated as such in the manuscript, not
glossed over. 3-fold spatial CV demonstrates whether the model transfers to
an unseen basin at all; it does not demonstrate fine-grained spatial
generalization the way a 10+ fold study would. Report it as "preliminary
spatial generalization check (3-fold LOBO)", not as equivalent evidence to
the prior LOSO study.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from floodai.evaluation.metrics import DataProvenance, EvaluationResult, evaluate
from floodai.features.governance import assert_no_forbidden_columns
from floodai.models.xgb_model import build_xgb_classifier, fit_with_validation
from floodai.training.imbalance import resample_training_only
from floodai.training.threshold import select_f1_optimal_threshold

logger = logging.getLogger("floodai.training.lobo")


def run_lobo_cv(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    basin_column: str,
    best_params: dict,
    early_stopping_rounds: int,
    smote_sampling_strategy: float,
    smote_k_neighbors_max: int,
    seed: int,
) -> list[EvaluationResult]:
    """
    For each basin, train on the other basins and evaluate on the held-out
    basin (data this fold's model never saw, basin-wise). Uses an internal
    85/15 split of the training basins' data for early stopping (mirrors the
    approach used for the original LOSO in the prior notebook).
    """
    basins = sorted(df[basin_column].unique())
    if len(basins) < 2:
        raise ValueError("LOBO requires at least 2 basins; got 1. Cannot leave one out of a singleton set.")

    assert_no_forbidden_columns(feature_columns)  # see features/governance.py — Year-leakage incident

    results: list[EvaluationResult] = []
    for held_out in basins:
        train_mask = df[basin_column] != held_out
        test_mask = df[basin_column] == held_out

        X_tr_full = df.loc[train_mask, feature_columns]
        y_tr_full = df.loc[train_mask, target_column]
        X_te = df.loc[test_mask, feature_columns]
        y_te = df.loc[test_mask, target_column]

        if y_te.sum() == 0:
            logger.warning("Basin '%s' has zero flood-positive days; skipping this LOBO fold.", held_out)
            continue

        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_full)
        X_te_scaled = scaler.transform(X_te)

        X_tr_res, y_tr_res = resample_training_only(
            X_tr_scaled, y_tr_full.values, smote_sampling_strategy, smote_k_neighbors_max, seed
        )

        split_idx = int(len(X_tr_res) * 0.85)
        model = build_xgb_classifier(best_params, early_stopping_rounds, seed)
        model = fit_with_validation(
            model, X_tr_res[:split_idx], y_tr_res[:split_idx],
            X_tr_res[split_idx:], y_tr_res[split_idx:],
        )

        proba_te = model.predict_proba(X_te_scaled)[:, 1]
        threshold = select_f1_optimal_threshold(y_tr_res[split_idx:],
                                                 model.predict_proba(X_tr_res[split_idx:])[:, 1])

        result = evaluate(
            y_te.values, proba_te, threshold=threshold,
            set_name=f"LOBO_held_out_{held_out}",
            provenance=DataProvenance.LOSO_HELD_OUT,
        )
        results.append(result)

    aucs = [r.roc_auc for r in results]
    logger.info(
        "LOBO complete: %d folds, mean AUC=%.4f +/- %.4f. "
        "NOTE: only %d folds — report as preliminary spatial generalization "
        "check, not equivalent to a larger-N LOSO study.",
        len(results), float(np.mean(aucs)), float(np.std(aucs)), len(results),
    )
    return results
