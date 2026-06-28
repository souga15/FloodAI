"""
Flood-event label sufficiency checks.

Why this exists: a real run of this pipeline had only 10 verified flood
events total across 3 basins x 8 years. None fell within the 2023-2024 test
window, so the held-out test set had zero positive samples — ROC-AUC was
mathematically undefined (NaN) and F1 was trivially 0.0000. That is not a
modeling failure; it is a data-sufficiency failure that should be caught
*before* spending compute on Optuna tuning, not discovered after.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("floodai.training.label_sufficiency")


class InsufficientLabelsError(Exception):
    """Raised when a split (or basin) has too few/zero positive labels to evaluate meaningfully."""


def check_split_has_positives(
    df: pd.DataFrame,
    date_col: str,
    label_col: str,
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    min_positives_per_split: int = 5,
) -> None:
    """
    Raises InsufficientLabelsError with a precise, actionable message if any
    split has fewer than `min_positives_per_split` positive labels. This is
    the check that would have caught the 10-total-events problem immediately
    after Step 5 (labelling), instead of after a 35-minute Optuna run.
    """
    year = df[date_col].dt.year
    splits = {"train": train_years, "val": val_years, "test": test_years}
    problems = []

    for split_name, years in splits.items():
        mask = year.isin(years)
        n_positive = int(df.loc[mask, label_col].sum())
        n_total = int(mask.sum())
        logger.info(
            "Split '%s' (years=%s): %d positive / %d total (%.3f%%)",
            split_name, years, n_positive, n_total,
            100 * n_positive / n_total if n_total else 0,
        )
        if n_positive < min_positives_per_split:
            problems.append(
                f"  - {split_name} (years {years}): only {n_positive} positive "
                f"labels (minimum required: {min_positives_per_split})"
            )

    if problems:
        raise InsufficientLabelsError(
            "One or more splits have too few positive (flood) labels to "
            "produce a meaningful evaluation:\n" + "\n".join(problems) +
            "\n\nThis is very likely a flood-event coverage problem, not a "
            "code bug: check whether data/flood_events_basins.csv actually "
            "contains verified events falling within each split's year "
            "range. A common cause is curating a small set of well-known "
            "major events that happen to cluster in certain years, leaving "
            "other splits (often the most recent test years) with none. "
            "Add more verified CWC/DFO/EM-DAT events before proceeding -- "
            "do not relax min_positives_per_split as a workaround, since a "
            "near-empty split produces an unreliable metric regardless of "
            "the threshold used here."
        )


def check_basin_has_positives(
    df: pd.DataFrame, basin_col: str, label_col: str, min_positives: int = 5
) -> dict[str, int]:
    """Per-basin positive-label counts, for sanity-checking before LOBO.
    Returns the counts dict; does not raise, since a basin with genuinely
    near-zero floods (e.g. Mahanadi as a documented hard case) is a valid,
    reportable finding rather than an error -- but logs a clear warning."""
    counts = df.groupby(basin_col)[label_col].sum().to_dict()
    for basin, n in counts.items():
        if n < min_positives:
            logger.warning(
                "Basin '%s' has only %d positive labels. LOBO results for "
                "this basin (whether held out or used as training data) "
                "will be unstable. If this is expected (e.g. a documented "
                "hard case), report it as such rather than treating the "
                "resulting metric as a precise estimate.",
                basin, n,
            )
    return {k: int(v) for k, v in counts.items()}
