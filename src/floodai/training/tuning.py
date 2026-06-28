"""
Optuna hyperparameter search for the XGBoost flood-occurrence model.

All search bounds come from config.yaml (model.optuna.search_space) — this
module contains zero hard-coded hyperparameter ranges, so changing the
search space never requires touching this file.
"""
from __future__ import annotations

import logging

import optuna
from sklearn.metrics import average_precision_score

from floodai.models.xgb_model import build_xgb_classifier, fit_with_validation

logger = logging.getLogger("floodai.training.tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    params = {}
    for name, spec in search_space.items():
        if "log" in spec and spec.get("log"):
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif isinstance(spec["low"], int) and isinstance(spec["high"], int):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
    return params


def run_optuna_search(
    X_train_resampled,
    y_train_resampled,
    X_val,
    y_val,
    search_space: dict,
    n_trials: int,
    early_stopping_rounds: int,
    seed: int,
) -> dict:
    """
    Returns the best hyperparameter dict found over n_trials, optimizing
    validation-set average precision (PR-AUC) — the right primary objective
    for a ~2-10% positive class rate, where ROC-AUC alone is too easy to
    inflate via the negative class.
    """

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, search_space)
        model = build_xgb_classifier(params, early_stopping_rounds, seed)
        model = fit_with_validation(model, X_train_resampled, y_train_resampled, X_val, y_val)
        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Optuna search complete: %d trials, best val PR-AUC=%.4f, best params=%s",
        n_trials, study.best_value, study.best_params,
    )
    return study.best_params
