"""
Tests for floodai.evaluation.label_diagnostics.

These tests construct two synthetic scenarios:
  - A basin where ALL labeled flood events occur during genuinely elevated
    rainfall (mimics a "real, rainfall-driven flood" basin, or explanation
    (a): the model legitimately learned a rainfall-correlated mechanism).
  - A basin where labeled events are deliberately rainfall-QUIET (mimics
    explanation (b): label curation/selection bias, or a genuinely
    non-rainfall-driven flood mechanism like a dam release).

The diagnostic must clearly separate these two cases on
`rainfall_percentile_vs_baseline` and `fraction_rainfall_coincident`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodai.evaluation.label_diagnostics import (
    compare_basins_rainfall_coincidence,
    compute_event_rainfall_context,
)


def _make_basin_df(basin_key: str, seed: int, n_days: int = 1000):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n_days, freq="D")
    rainfall = rng.gamma(2.0, 8.0, n_days)  # baseline rainfall distribution
    df = pd.DataFrame({
        "Date": dates, "basin_key": basin_key,
        "Rainfall_7Day_mm": rainfall, "Flood_Occurred": 0,
    })
    return df


class TestRainfallCoincidentEvents:
    def test_rainfall_heavy_events_score_high_percentile(self):
        """Explanation (a) scenario: events deliberately placed during
        rainfall spikes."""
        df = _make_basin_df("test_basin", seed=1)
        # Inject a rainfall spike on days 100-105 and mark that as the event window
        df.loc[100:105, "Rainfall_7Day_mm"] = 400.0  # far above baseline gamma(2,8) ~ mean 16
        df.loc[100:105, "Flood_Occurred"] = 1

        events = pd.DataFrame([{
            "basin_key": "test_basin",
            "Start": df.loc[100, "Date"], "End": df.loc[105, "Date"],
            "Severity": "High", "Source": "CWC",
        }])

        result = compute_event_rainfall_context(df, events, basin_key="test_basin")
        assert len(result) == 1
        assert result.iloc[0]["rainfall_percentile_vs_baseline"] > 90, (
            "A deliberately rainfall-heavy event should score a high percentile "
            "relative to the basin's non-flood baseline."
        )

    def test_rainfall_quiet_events_score_low_percentile(self):
        """Explanation (b) scenario: events placed during unremarkable
        (or below-baseline) rainfall -- e.g. a pure reservoir-release flood."""
        df = _make_basin_df("test_basin", seed=2)
        df.loc[200:205, "Rainfall_7Day_mm"] = 2.0  # far BELOW baseline
        df.loc[200:205, "Flood_Occurred"] = 1

        events = pd.DataFrame([{
            "basin_key": "test_basin",
            "Start": df.loc[200, "Date"], "End": df.loc[205, "Date"],
            "Severity": "High", "Source": "CWC",
        }])

        result = compute_event_rainfall_context(df, events, basin_key="test_basin")
        assert result.iloc[0]["rainfall_percentile_vs_baseline"] < 10, (
            "A deliberately rainfall-quiet event should score a low percentile "
            "-- this is the signature of a non-rainfall-driven (e.g. reservoir "
            "release) flood, or of label-curation bias if it's unexpectedly common."
        )

    def test_cross_basin_comparison_separates_the_two_patterns(self):
        """The comparison summary must show a clear, large difference in
        fraction_rainfall_coincident between a rainfall-driven basin and a
        rainfall-quiet (e.g. reservoir-driven) basin."""
        df_rainy = _make_basin_df("rainy_basin", seed=3)
        for start in [50, 150, 250, 350]:
            df_rainy.loc[start:start + 4, "Rainfall_7Day_mm"] = 350.0
            df_rainy.loc[start:start + 4, "Flood_Occurred"] = 1

        df_quiet = _make_basin_df("quiet_basin", seed=4)
        for start in [60, 160, 260, 360]:
            df_quiet.loc[start:start + 4, "Rainfall_7Day_mm"] = 1.0
            df_quiet.loc[start:start + 4, "Flood_Occurred"] = 1

        combined = pd.concat([df_rainy, df_quiet], ignore_index=True)

        events = pd.DataFrame([
            {"basin_key": "rainy_basin", "Start": df_rainy.loc[s, "Date"], "End": df_rainy.loc[s + 4, "Date"],
             "Severity": "High", "Source": "CWC"}
            for s in [50, 150, 250, 350]
        ] + [
            {"basin_key": "quiet_basin", "Start": df_quiet.loc[s, "Date"], "End": df_quiet.loc[s + 4, "Date"],
             "Severity": "High", "Source": "CWC"}
            for s in [60, 160, 260, 360]
        ])

        summary = compare_basins_rainfall_coincidence(combined, events, basin_keys=["rainy_basin", "quiet_basin"])
        rainy_frac = summary.loc[summary["basin"] == "rainy_basin", "fraction_rainfall_coincident"].iloc[0]
        quiet_frac = summary.loc[summary["basin"] == "quiet_basin", "fraction_rainfall_coincident"].iloc[0]

        assert rainy_frac == pytest.approx(1.0)
        assert quiet_frac == pytest.approx(0.0)

    def test_raises_on_basin_with_no_events(self):
        df = _make_basin_df("empty_basin", seed=5)
        events = pd.DataFrame([{
            "basin_key": "other_basin", "Start": df.loc[0, "Date"], "End": df.loc[5, "Date"],
            "Severity": "Low", "Source": "DFO",
        }])
        with pytest.raises(ValueError, match="No flood events"):
            compute_event_rainfall_context(df, events, basin_key="empty_basin")
