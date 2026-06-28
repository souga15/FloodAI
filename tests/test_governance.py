"""
Tests for floodai.features.governance — regression tests for the real
incident where a notebook's manually-maintained `exclude_cols` set omitted
`Year`, letting it leak into the model and producing a spurious
LOBO AUC=1.000 / Recall=1.000 across all three basins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodai.features.governance import (
    NEVER_FEATURES,
    FeatureGovernanceError,
    assert_no_forbidden_columns,
    select_model_features,
)
from floodai.features.pipeline import add_rainfall_window_features, add_temporal_features


def _featured_df():
    dates = pd.date_range("2017-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Date": dates, "point_id": "p1", "basin_key": "ganga_bihar",
        "Rainfall_mm": rng.gamma(1, 5, len(dates)),
    })
    df = add_temporal_features(df)
    df = add_rainfall_window_features(df, group_col="point_id")
    return df


class TestYearLeakageRegression:
    def test_year_is_never_in_selected_features(self):
        """Direct regression test for the incident: Year must never appear
        in the output of select_model_features, regardless of which groups
        are requested."""
        df = _featured_df()
        cols = select_model_features(df)
        assert "Year" not in cols, (
            "REGRESSION: 'Year' appeared in selected model features. This "
            "is the exact bug that caused spurious LOBO AUC=1.000 with only "
            "10 sparse flood events across 8 years — the model could use "
            "raw Year as a lookup key for which years happen to contain a "
            "labeled flood, instead of learning from rainfall/terrain."
        )

    def test_year_is_in_never_features_set(self):
        assert "Year" in NEVER_FEATURES

    def test_seasonal_encodings_are_kept_unlike_year(self):
        """Month/Day_of_Year and their sin/cos encodings legitimately encode
        seasonality (a real causal proxy) and must NOT be excluded the way
        Year is -- this test guards against someone overcorrecting and
        stripping all temporal information."""
        df = _featured_df()
        cols = select_model_features(df, groups=["temporal"])
        for expected in ["Month", "Day_of_Year", "Month_Sin", "Month_Cos", "Is_Monsoon_Season"]:
            assert expected in cols, f"Expected seasonal feature '{expected}' was incorrectly excluded"

    def test_assert_no_forbidden_columns_catches_manually_built_list(self):
        """Simulates the exact notebook pattern that caused the incident:
        a hand-rolled exclude_cols set that forgot 'Year'. The defensive
        assert function must catch this even when select_model_features()
        was bypassed entirely."""
        df = _featured_df()
        exclude_cols = {"Date", "point_id", "basin_key", "lat", "lon", "Flood_Occurred"}  # Year missing, as in the real incident
        manually_built_feature_cols = [c for c in df.columns if c not in exclude_cols]

        assert "Year" in manually_built_feature_cols  # confirms this reproduces the bug
        with pytest.raises(FeatureGovernanceError, match="Year"):
            assert_no_forbidden_columns(manually_built_feature_cols)

    def test_clean_feature_list_passes_assertion(self):
        df = _featured_df()
        cols = select_model_features(df)
        assert_no_forbidden_columns(cols)  # must not raise

    def test_unknown_group_raises(self):
        df = _featured_df()
        with pytest.raises(FeatureGovernanceError, match="Unknown feature group"):
            select_model_features(df, groups=["nonexistent_group"])
