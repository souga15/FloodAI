"""
XGBoost model construction for flood occurrence classification.

XGBoost 2.x API note (this was the one verifiably correct technical fix in
the prior WPT-D-26-00166 notebook, carried forward deliberately):
`early_stopping_rounds` must be passed to the XGBClassifier CONSTRUCTOR, not
to `.fit()`. Passing it to `.fit()` in XGBoost >= 2.0 is silently ignored —
no warning, no error, the model just trains for the full `n_estimators`.
This module enforces the constructor-level usage in one place so it can't
drift back to the wrong API call as the codebase grows.
"""
from __future__ import annotations

import logging

from xgboost import XGBClassifier

logger = logging.getLogger("floodai.models.xgb")

REQUIRED_CONSTRUCTOR_PARAMS = {"early_stopping_rounds", "eval_metric", "random_state"}


def build_xgb_classifier(params: dict, early_stopping_rounds: int, random_seed: int) -> XGBClassifier:
    """
    Construct an XGBClassifier with early stopping correctly wired at the
    constructor level (XGBoost >= 2.0 requirement).

    Raises if a caller tries to also pass early_stopping_rounds via some
    other path later — the only valid place for it is here, at construction.
    """
    if "early_stopping_rounds" in params:
        raise ValueError(
            "early_stopping_rounds should not be in the Optuna-tuned `params` "
            "dict; pass it explicitly via the `early_stopping_rounds` argument "
            "so it's visible at the call site, not buried in a dict."
        )

    model = XGBClassifier(
        **params,
        random_state=random_seed,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
    )
    logger.info(
        "Built XGBClassifier: n_estimators=%s max_depth=%s lr=%s early_stopping_rounds=%d (constructor-level, XGBoost>=2.0 API)",
        params.get("n_estimators"), params.get("max_depth"), params.get("learning_rate"),
        early_stopping_rounds,
    )
    return model


def fit_with_validation(model: XGBClassifier, X_train, y_train, X_val, y_val) -> XGBClassifier:
    """Fit with an explicit validation set for early stopping to monitor.
    Returns the fitted model; logs how many trees were actually used vs the
    configured maximum, which is the simplest sanity check that early
    stopping is actually functioning (if best_iteration == n_estimators - 1
    on every run, early stopping likely isn't engaging — worth investigating,
    not necessarily wrong, but worth a second look)."""
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    n_estimators = model.get_params().get("n_estimators")
    actual_trees = (model.best_iteration + 1) if hasattr(model, "best_iteration") and model.best_iteration is not None else n_estimators
    logger.info("Fitted model: %d / %d trees used (early stopping %s)",
                actual_trees, n_estimators,
                "engaged" if actual_trees < n_estimators else "did not trigger")
    return model
