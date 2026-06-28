"""
Feature column governance.

This module exists because of a real incident: in an early run of this
pipeline, a notebook-level `exclude_cols` set was used to select model
features from a feature-engineered DataFrame, and it omitted `Year` —
which `add_temporal_features()` adds as a literal integer column. Because
flood events were sparse (10 verified events total) and clustered in
specific years, `Year` let XGBoost effectively memorize "which years have
labeled floods" rather than learn rainfall-driven flood physics. This
produced Leave-One-Basin-Out AUC=1.000 / Recall=1.000 simultaneously across
three basins with different flood-generation mechanisms — a result that
looked like outstanding generalization but was actually near-total leakage.

The fix here is structural, not a corrected comment: feature selection is no
longer "everything except a manually maintained exclude list" computed ad
hoc in notebook code. It is an explicit ALLOWLIST defined once in this
module, reviewed alongside floodai.features.pipeline so the two can never
drift apart silently.

`Year` itself is deliberately never eligible to be a model feature in this
project: it has no value as a flood-occurrence predictor (a date's *year*
does not cause floods; rainfall, soil moisture, and terrain do), and because
flood event labels are temporally sparse relative to the daily record, any
column that can identify *which calendar year* a row belongs to creates a
shortcut around the actual prediction task. `Month`, `Day_of_Year`,
`Week_of_Year`, and their sine/cosine encodings ARE kept: these encode
*seasonality* (a real causal proxy for monsoon timing), not which specific
year, and rotate every 12 months rather than monotonically identifying a
unique year.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("floodai.features.governance")

# Columns that must NEVER be passed to a model, under any configuration.
# A column landing in this set is structural metadata (identifiers, raw
# dates) or a documented leakage risk (Year) — not a candidate feature.
NEVER_FEATURES = frozenset({
    "Date", "point_id", "basin_key", "lat", "lon", "name", "source",
    "Region", "Basin", "Flood_Occurred",
    "Year",  # see module docstring: sparse event labels + raw Year = leakage shortcut
})

# Explicit allowlist of feature groups produced by floodai.features.pipeline.
# Adding a new engineered feature in pipeline.py does NOT automatically make
# it model-eligible — it must be added here deliberately, so a reviewer can
# see every feature the model is allowed to use in one place.
ALLOWED_FEATURE_GROUPS: dict[str, list[str]] = {
    "temporal": [
        "Month", "Day_of_Year", "Week_of_Year",
        "Is_Monsoon_Season", "Is_Peak_Monsoon", "Is_Pre_Monsoon", "Is_Post_Monsoon",
        "Month_Sin", "Month_Cos", "Day_of_Year_Sin", "Day_of_Year_Cos",
    ],
    "rainfall_current": ["Rainfall_mm"],
    "rainfall_windows": [
        "Rainfall_3Day_mm", "Rainfall_7Day_mm", "Rainfall_14Day_mm",
        "Rainfall_30Day_mm", "Rainfall_60Day_mm",
        "Rainfall_7Day_Avg", "Rainfall_7Day_Max", "Rainfall_7Day_Std", "Rainfall_30Day_Std",
        "Heavy_Rain_Days_7D", "Extreme_Rain_Days_7D",
        "Consecutive_Dry_Days", "Soil_Moisture_Proxy", "Rainfall_Acceleration",
    ],
    "terrain_physics": [
        "Elevation_m", "Curve_Number", "TWI", "CN_Runoff_Q",
    ],
    "interaction": [
        "Elevation_Rain_Ratio", "Elevation_Rain30_Ratio",
        "Monsoon_Rain_Interaction", "Peak_Monsoon_Rain",
        "Humidity_Temp_Product", "Rain_Humidity_Product",
        "Soil_Monsoon_Interaction", "Low_Elev_Heavy_Rain",
    ],
}


class FeatureGovernanceError(Exception):
    """Raised when feature selection would violate the allowlist or include a forbidden column."""


def select_model_features(df: pd.DataFrame, groups: list[str] | None = None) -> list[str]:
    """
    Return the list of columns from `df` that are both (a) present and
    (b) on the allowlist for the requested feature groups. Defaults to all
    groups. This is the ONLY sanctioned way to derive `feature_cols` in this
    project — do not reconstruct it via `[c for c in df.columns if c not in
    exclude_set]` in notebook code, which is exactly the pattern that let
    `Year` slip through previously.

    Raises FeatureGovernanceError if any column on the resulting list is
    also in NEVER_FEATURES (defensive double-check; should be unreachable
    given the allowlist construction, but cheap to verify at runtime).
    """
    if groups is None:
        groups = list(ALLOWED_FEATURE_GROUPS.keys())

    unknown_groups = set(groups) - set(ALLOWED_FEATURE_GROUPS.keys())
    if unknown_groups:
        raise FeatureGovernanceError(f"Unknown feature group(s): {unknown_groups}")

    candidate_cols: list[str] = []
    for g in groups:
        candidate_cols.extend(ALLOWED_FEATURE_GROUPS[g])

    selected = [c for c in candidate_cols if c in df.columns]
    missing = [c for c in candidate_cols if c not in df.columns]
    if missing:
        logger.warning(
            "select_model_features: %d allowlisted columns not present in "
            "this DataFrame (likely not yet computed, e.g. terrain joins "
            "pending real SRTM/CN data): %s",
            len(missing), missing,
        )

    forbidden_hits = set(selected) & NEVER_FEATURES
    if forbidden_hits:
        raise FeatureGovernanceError(
            f"INTERNAL ERROR: {forbidden_hits} are in both the allowlist and "
            f"NEVER_FEATURES. This should be impossible — fix ALLOWED_FEATURE_GROUPS."
        )

    logger.info("select_model_features: selected %d columns from groups %s", len(selected), groups)
    return selected


def assert_no_forbidden_columns(feature_cols: list[str]) -> None:
    """
    Defensive check to call right before model.fit() / predict(), even if
    feature_cols came from somewhere other than select_model_features()
    (e.g. a LOBO fold or an older notebook cell). Raises loudly rather than
    training on a forbidden column.
    """
    hits = set(feature_cols) & NEVER_FEATURES
    if hits:
        raise FeatureGovernanceError(
            f"Refusing to train/predict: feature_cols contains forbidden "
            f"column(s) {hits}. This is the exact failure mode that produced "
            f"a spurious LOBO AUC=1.000 in an earlier run (raw 'Year' column "
            f"leaking which calendar year sparse flood labels fall in). "
            f"Use floodai.features.governance.select_model_features() to "
            f"build feature_cols instead of a manually maintained exclude set."
        )
