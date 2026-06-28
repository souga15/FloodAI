"""Tests for floodai.training.label_sufficiency — reproduces the real
incident (10 total flood events, zero falling in the 2023-2024 test window,
producing NaN ROC-AUC and F1=0.0000) and proves it is now caught early."""
from __future__ import annotations

import pandas as pd
import pytest

from floodai.training.label_sufficiency import (
    InsufficientLabelsError,
    check_basin_has_positives,
    check_split_has_positives,
)


def _df_with_events_only_in_early_years():
    """Mirrors the real incident: flood events exist for 2017-2020 but none
    for 2023-2024 (the test window)."""
    dates = pd.date_range("2017-01-01", "2024-12-31", freq="D")
    df = pd.DataFrame({"Date": dates, "Flood_Occurred": 0})
    # Inject positives only in 2019 (mimics "10 events, clustered early")
    df.loc[df["Date"].dt.year == 2019, "Flood_Occurred"] = (
        df.loc[df["Date"].dt.year == 2019, "Date"].dt.dayofyear % 30 == 0
    ).astype(int)
    return df


class TestLabelSufficiencyRegression:
    def test_zero_test_positives_raises_before_training(self):
        """Direct regression test: this exact data shape previously reached
        Optuna/XGBoost training and only failed silently at evaluation time
        (NaN AUC, F1=0.0000, 35 minutes of wasted compute). It must now raise
        immediately after labelling."""
        df = _df_with_events_only_in_early_years()
        with pytest.raises(InsufficientLabelsError, match="test"):
            check_split_has_positives(
                df, date_col="Date", label_col="Flood_Occurred",
                train_years=[2017, 2018, 2019, 2020],
                val_years=[2021, 2022],
                test_years=[2023, 2024],
                min_positives_per_split=5,
            )

    def test_sufficient_positives_in_all_splits_passes(self):
        dates = pd.date_range("2017-01-01", "2024-12-31", freq="D")
        df = pd.DataFrame({"Date": dates, "Flood_Occurred": 0})
        for yr in [2018, 2021, 2024]:
            mask = df["Date"].dt.year == yr
            df.loc[mask, "Flood_Occurred"] = (df.loc[mask, "Date"].dt.dayofyear % 10 == 0).astype(int)
        check_split_has_positives(
            df, date_col="Date", label_col="Flood_Occurred",
            train_years=[2017, 2018, 2019, 2020], val_years=[2021, 2022], test_years=[2023, 2024],
            min_positives_per_split=5,
        )  # must not raise

    def test_basin_check_does_not_raise_for_documented_hard_case(self):
        """A basin with genuinely near-zero floods (e.g. Mahanadi) is a valid
        finding, not an error -- this check warns but does not block."""
        df = pd.DataFrame({
            "basin_key": ["mahanadi_odisha"] * 100 + ["ganga_bihar"] * 100,
            "Flood_Occurred": [0] * 100 + [1] * 20 + [0] * 80,
        })
        counts = check_basin_has_positives(df, basin_col="basin_key", label_col="Flood_Occurred")
        assert counts["mahanadi_odisha"] == 0
        assert counts["ganga_bihar"] == 20
