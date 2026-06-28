"""
Leakage-safe feature engineering for daily station/point-level flood
occurrence modelling.

Provenance: the rolling-window logic here is carried forward from
WPT-D-26-00166's `engineer_features_no_leakage()`, which was reviewed and
found sound — every rolling aggregate is `.shift(1)`'d before use, so no
feature at row t uses information from day t itself or later. That property
is what tests/test_features.py checks automatically; do not modify a
rolling-feature function without re-running those tests.

What changed from the original: functions are now small, single-purpose,
and independently testable, instead of one 150-line function building
everything inline. This matters because a 150-line function is exactly
where a missing `.shift(1)` hides.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("floodai.features")

RAINFALL_WINDOWS_DAYS = (3, 7, 14, 30, 60)
HEAVY_RAIN_THRESHOLD_MM = 50.0
EXTREME_RAIN_THRESHOLD_MM = 100.0
DRY_DAY_THRESHOLD_MM = 5.0


def add_temporal_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Calendar/seasonal features. These use only the date itself — never leakage-prone."""
    df = df.copy()
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day_of_Year"] = df[date_col].dt.dayofyear
    df["Week_of_Year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["Is_Monsoon_Season"] = df["Month"].isin([6, 7, 8, 9]).astype(int)
    df["Is_Peak_Monsoon"] = df["Month"].isin([7, 8]).astype(int)
    df["Is_Pre_Monsoon"] = df["Month"].isin([4, 5]).astype(int)
    df["Is_Post_Monsoon"] = df["Month"].isin([10, 11]).astype(int)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Day_of_Year_Sin"] = np.sin(2 * np.pi * df["Day_of_Year"] / 365)
    df["Day_of_Year_Cos"] = np.cos(2 * np.pi * df["Day_of_Year"] / 365)
    return df


def add_rainfall_window_features(
    df: pd.DataFrame, group_col: str, rain_col: str = "Rainfall_mm"
) -> pd.DataFrame:
    """
    Cumulative and rolling-statistic rainfall features per group (point/station).
    Every `.rolling(...)` call is immediately followed by `.shift(1)` — this is
    the no-leakage contract. df must already be sorted by [group_col, Date].
    """
    df = df.copy()
    for group, idx in df.groupby(group_col).groups.items():
        rain = df.loc[idx, rain_col]

        for days in RAINFALL_WINDOWS_DAYS:
            df.loc[idx, f"Rainfall_{days}Day_mm"] = (
                rain.rolling(days, min_periods=1).sum().shift(1).fillna(0)
            )

        df.loc[idx, "Rainfall_7Day_Avg"] = rain.rolling(7, min_periods=1).mean().shift(1).fillna(0)
        df.loc[idx, "Rainfall_7Day_Max"] = rain.rolling(7, min_periods=1).max().shift(1).fillna(0)
        df.loc[idx, "Rainfall_7Day_Std"] = rain.rolling(7, min_periods=2).std().shift(1).fillna(0)
        df.loc[idx, "Rainfall_30Day_Std"] = rain.rolling(30, min_periods=2).std().shift(1).fillna(0)

        heavy = (rain > HEAVY_RAIN_THRESHOLD_MM).astype(int)
        df.loc[idx, "Heavy_Rain_Days_7D"] = heavy.rolling(7, min_periods=1).sum().shift(1).fillna(0)

        extreme = (rain > EXTREME_RAIN_THRESHOLD_MM).astype(int)
        df.loc[idx, "Extreme_Rain_Days_7D"] = extreme.rolling(7, min_periods=1).sum().shift(1).fillna(0)

        dry = (rain < DRY_DAY_THRESHOLD_MM).astype(int)
        df.loc[idx, "Consecutive_Dry_Days"] = dry.rolling(14, min_periods=1).sum().shift(1).fillna(0)

        df.loc[idx, "Soil_Moisture_Proxy"] = rain.ewm(span=14).mean().shift(1).fillna(0)

        rain_3d = rain.rolling(3, min_periods=1).sum()
        rain_7d = rain.rolling(7, min_periods=1).sum()
        df.loc[idx, "Rainfall_Acceleration"] = ((rain_3d - rain_7d / 7 * 3).shift(1).fillna(0))

    return df


def add_scs_cn_runoff(
    df: pd.DataFrame, rain_7day_col: str = "Rainfall_7Day_mm", curve_number_col: str = "Curve_Number"
) -> pd.DataFrame:
    """
    SCS Curve Number runoff (USDA TR-55 / IS SP-30, AMC-II):
        S  = 1000/CN - 10                       (potential max retention, inches)
        Ia = 0.2 * S                             (initial abstraction)
        Q  = (P - Ia)^2 / (P - Ia + S)  for P > Ia, else 0
    P is taken as the no-leakage 7-day cumulative rainfall, converted mm->inches.
    """
    df = df.copy()
    p_in = df[rain_7day_col] / 25.4
    s = (1000.0 / df[curve_number_col].clip(lower=1)) - 10
    ia = 0.2 * s
    excess = (p_in - ia).clip(lower=0)
    runoff_in = (excess ** 2) / (excess + s + 1e-6)
    df["CN_Runoff_Q"] = runoff_in * 25.4
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-terms between rainfall, terrain, and seasonality. Each term must
    already exist as a no-leakage column before this runs (call order matters
    — see features/pipeline.py for the enforced order)."""
    df = df.copy()
    required = [
        "Rainfall_7Day_mm", "Rainfall_30Day_mm", "Elevation_m",
        "Is_Monsoon_Season", "Is_Peak_Monsoon", "Humidity_pct", "Temperature_C",
        "Soil_Moisture_Proxy",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"add_interaction_features requires columns {missing} to exist first. "
            "Check pipeline call order in features/pipeline.py."
        )

    df["Elevation_Rain_Ratio"] = df["Rainfall_7Day_mm"] / (df["Elevation_m"] + 1)
    df["Elevation_Rain30_Ratio"] = df["Rainfall_30Day_mm"] / (df["Elevation_m"] + 1)
    df["Monsoon_Rain_Interaction"] = df["Is_Monsoon_Season"] * df["Rainfall_7Day_mm"]
    df["Peak_Monsoon_Rain"] = df["Is_Peak_Monsoon"] * df["Rainfall_7Day_mm"]
    df["Humidity_Temp_Product"] = df["Humidity_pct"] * df["Temperature_C"] / 100
    df["Rain_Humidity_Product"] = df["Rainfall_7Day_mm"] * df["Humidity_pct"] / 100
    df["Soil_Monsoon_Interaction"] = df["Soil_Moisture_Proxy"] * df["Is_Monsoon_Season"]
    df["Low_Elev_Heavy_Rain"] = (df["Elevation_m"] < 50).astype(int) * df["Rainfall_7Day_mm"]
    return df
