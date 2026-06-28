"""
Tests for floodai.features.pipeline — specifically proving the no-leakage
property: a rolling feature computed at row t must be IDENTICAL whether or
not rows after t exist in the input at all. If future rows could change a
past row's feature value, that is leakage by definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodai.features.pipeline import (
    add_rainfall_window_features,
    add_scs_cn_runoff,
    add_temporal_features,
)


def _make_synthetic_series(n_days: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rain = rng.gamma(shape=1.0, scale=8.0, size=n_days)
    return pd.DataFrame({"Date": dates, "Region": "TestPoint", "Rainfall_mm": rain})


class TestNoLeakage:
    def test_rolling_feature_unaffected_by_future_truncation(self):
        """The defining leakage test: truncate the series after day t and
        recompute. Feature values for all days <= t must be byte-identical
        to the values computed on the full series."""
        df_full = _make_synthetic_series(n_days=100)
        df_full = df_full.sort_values(["Region", "Date"]).reset_index(drop=True)
        full_featured = add_rainfall_window_features(df_full, group_col="Region")

        cutoff = 60
        df_truncated = df_full.iloc[: cutoff + 1].copy()
        truncated_featured = add_rainfall_window_features(df_truncated, group_col="Region")

        feature_cols = [c for c in full_featured.columns if c.startswith("Rainfall_") or "Dry" in c or "Soil_Moisture" in c]
        for col in feature_cols:
            full_vals = full_featured.loc[: cutoff, col].to_numpy()
            trunc_vals = truncated_featured.loc[: cutoff, col].to_numpy()
            np.testing.assert_allclose(
                full_vals, trunc_vals, rtol=1e-9, atol=1e-9,
                err_msg=(
                    f"LEAKAGE DETECTED in column '{col}': values for rows <= cutoff "
                    f"changed depending on whether future rows exist. A rolling "
                    f"feature must not depend on data the model wouldn't have had "
                    f"at prediction time."
                ),
            )

    def test_row_zero_uses_no_prior_information(self):
        """The very first row of a series has no history. Shifted rolling
        features at row 0 must equal the fillna default (0), not some
        leaked statistic."""
        df = _make_synthetic_series(n_days=30)
        featured = add_rainfall_window_features(df, group_col="Region")
        first_row = featured.iloc[0]
        for days in (3, 7, 14, 30, 60):
            assert first_row[f"Rainfall_{days}Day_mm"] == 0.0, (
                f"Row 0 should have zero cumulative rainfall (no prior data), "
                f"got {first_row[f'Rainfall_{days}Day_mm']}"
            )

    def test_multiple_groups_do_not_leak_across_each_other(self):
        """Region A's rolling features must not be influenced by Region B's data."""
        df_a = _make_synthetic_series(n_days=50, seed=1)
        df_a["Region"] = "A"
        df_b = _make_synthetic_series(n_days=50, seed=2)
        df_b["Region"] = "B"
        df_b["Rainfall_mm"] = df_b["Rainfall_mm"] + 1000  # obviously distinct

        combined = pd.concat([df_a, df_b], ignore_index=True).sort_values(["Region", "Date"]).reset_index(drop=True)
        featured = add_rainfall_window_features(combined, group_col="Region")

        a_only_featured = add_rainfall_window_features(df_a.copy(), group_col="Region")
        a_rows_in_combined = featured[featured["Region"] == "A"].reset_index(drop=True)

        np.testing.assert_allclose(
            a_rows_in_combined["Rainfall_7Day_mm"].to_numpy(),
            a_only_featured["Rainfall_7Day_mm"].to_numpy(),
            err_msg="Region A's features changed when Region B was present in the same DataFrame — cross-group leakage.",
        )


class TestTemporalFeatures:
    def test_monsoon_flags_are_mutually_consistent(self):
        df = _make_synthetic_series(n_days=400)
        featured = add_temporal_features(df)
        # Peak monsoon (Jul/Aug) must be a subset of monsoon season (Jun-Sep)
        peak_rows = featured[featured["Is_Peak_Monsoon"] == 1]
        assert (peak_rows["Is_Monsoon_Season"] == 1).all(), "Peak monsoon rows must also be flagged as monsoon season"

    def test_no_nan_in_seasonal_encodings(self):
        df = _make_synthetic_series(n_days=10)
        featured = add_temporal_features(df)
        for col in ["Month_Sin", "Month_Cos", "Day_of_Year_Sin", "Day_of_Year_Cos"]:
            assert featured[col].notna().all(), f"{col} contains NaN — check Date parsing upstream"


class TestSCSRunoff:
    def test_runoff_is_nonnegative(self):
        df = pd.DataFrame({
            "Rainfall_7Day_mm": [0, 10, 50, 200, 500],
            "Curve_Number": [75, 75, 75, 75, 75],
        })
        out = add_scs_cn_runoff(df)
        assert (out["CN_Runoff_Q"] >= 0).all(), "SCS-CN runoff must never be negative"

    def test_higher_curve_number_increases_runoff_for_same_rainfall(self):
        df = pd.DataFrame({
            "Rainfall_7Day_mm": [100.0, 100.0],
            "Curve_Number": [60.0, 90.0],
        })
        out = add_scs_cn_runoff(df)
        assert out["CN_Runoff_Q"].iloc[1] > out["CN_Runoff_Q"].iloc[0], (
            "Higher CN (more impervious) should produce more runoff for identical rainfall"
        )

    def test_zero_rainfall_gives_zero_runoff(self):
        df = pd.DataFrame({"Rainfall_7Day_mm": [0.0], "Curve_Number": [80.0]})
        out = add_scs_cn_runoff(df)
        assert out["CN_Runoff_Q"].iloc[0] == pytest.approx(0.0, abs=1e-6)
