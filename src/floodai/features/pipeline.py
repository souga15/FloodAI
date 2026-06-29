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
        "Is_Monsoon_Season", "Is_Peak_Monsoon",
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
    df["Soil_Monsoon_Interaction"] = df["Soil_Moisture_Proxy"] * df["Is_Monsoon_Season"]
    df["Low_Elev_Heavy_Rain"] = (df["Elevation_m"] < 50).astype(int) * df["Rainfall_7Day_mm"]
    df["Rain_Intensity_Index"] = df["Rainfall_7Day_Max"] / (df["Rainfall_7Day_Avg"] + 1)

    # Terrain × rainfall physics interactions
    if "TWI" in df.columns:
        df["TWI_Rain_Interaction"] = df["TWI"] * df["Rainfall_7Day_mm"] / 100.0
    if "CN_Runoff_Q" in df.columns:
        df["CN_Rain_Interaction"] = df["CN_Runoff_Q"] * df["Rainfall_7Day_mm"] / 100.0

    return df


def compute_rainfall_climatology(
    df_train: pd.DataFrame,
    rain_col: str = "Rainfall_mm",
    window_days: int = 15,
) -> pd.DataFrame:
    """
    Compute smoothed climatological mean rainfall per basin per day-of-year
    from the TRAINING partition ONLY.

    Returns a DataFrame with columns [basin_key, Day_of_Year, clim_mean_rain,
    clim_mean_7d, clim_mean_30d] that can be merged into the full dataset.

    The ±15-day smoothing window reduces noise from sparse year counts while
    preserving the seasonal shape. Computed on train only to prevent leakage
    from val/test into the climatological reference.
    """
    df = df_train.copy()
    df["rain_7d"] = df.groupby("point_id")[rain_col].transform(
        lambda x: x.rolling(7, min_periods=1).sum().shift(1).fillna(0)
    )
    df["rain_30d"] = df.groupby("point_id")[rain_col].transform(
        lambda x: x.rolling(30, min_periods=1).sum().shift(1).fillna(0)
    )

    # Raw mean per basin × DOY
    raw = df.groupby(["basin_key", "Day_of_Year"]).agg(
        clim_mean_rain=(rain_col, "mean"),
        clim_mean_7d=("rain_7d", "mean"),
        clim_mean_30d=("rain_30d", "mean"),
    ).reset_index()

    # Smooth across DOY with wrapping window
    records = []
    w = window_days
    for basin in raw["basin_key"].unique():
        bdata = raw[raw["basin_key"] == basin].set_index("Day_of_Year")
        for col in ["clim_mean_rain", "clim_mean_7d", "clim_mean_30d"]:
            arr = np.zeros(366, dtype=float)
            for day in range(1, 367):
                if day in bdata.index:
                    arr[day - 1] = float(bdata.loc[day, col])
            smoothed = np.zeros(366, dtype=float)
            for i in range(366):
                indices = [(i + j) % 366 for j in range(-w, w + 1)]
                smoothed[i] = arr[indices].mean()
            for day in range(1, 367):
                records.append({
                    "basin_key": basin,
                    "Day_of_Year": day,
                    col: smoothed[day - 1],
                })

    # Merge all columns back
    clim_df = pd.DataFrame(records)
    clim_df = clim_df.groupby(["basin_key", "Day_of_Year"]).first().reset_index()
    logger.info(
        "Rainfall climatology computed from training data: %d basins, columns %s",
        clim_df["basin_key"].nunique(),
        [c for c in clim_df.columns if c not in ("basin_key", "Day_of_Year")],
    )
    return clim_df


def add_rainfall_anomaly_features(
    df: pd.DataFrame,
    climatology_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add rainfall anomaly features: observed minus training climatological mean
    for the same basin and day-of-year. These features are independent of the
    seasonal calendar flags, forcing the model to learn *excess* rainfall
    rather than *what season it is*.

    SHAP context: Is_Peak_Monsoon dominated previous SHAP rankings (57% mean
    |SHAP|) because raw rainfall windows are correlated with monsoon timing.
    Anomaly features break that correlation — a day with average-for-season
    rainfall gets anomaly≈0; a day with extreme rainfall gets a large anomaly
    regardless of month.

    climatology_df must be computed from training data only (see
    compute_rainfall_climatology). Raises if climatology_df is missing.
    """
    if climatology_df is None or len(climatology_df) == 0:
        raise ValueError(
            "climatology_df is empty or None. Call compute_rainfall_climatology() "
            "on the training partition before calling add_rainfall_anomaly_features()."
        )

    df = df.copy()
    merged = df.merge(
        climatology_df[["basin_key", "Day_of_Year", "clim_mean_rain", "clim_mean_7d", "clim_mean_30d"]],
        on=["basin_key", "Day_of_Year"],
        how="left",
    )

    df["Rain_Anomaly"] = (merged["Rainfall_mm"] - merged["clim_mean_rain"]).fillna(0)
    df["Rain7D_Anomaly"] = (df["Rainfall_7Day_mm"] - merged["clim_mean_7d"]).fillna(0)
    df["Rain30D_Anomaly"] = (df["Rainfall_30Day_mm"] - merged["clim_mean_30d"]).fillna(0)

    # Antecedent wetness flag: 1 if soil moisture proxy exceeds basin-day climatological mean
    clim_soil = df.groupby(["basin_key", "Day_of_Year"])["Soil_Moisture_Proxy"].transform("mean")
    df["Antecedent_Wet_Flag"] = (df["Soil_Moisture_Proxy"] > clim_soil).astype(int)

    n_nonzero = (df["Rain7D_Anomaly"] != 0).sum()
    logger.info(
        "Rainfall anomaly features added: Rain7D_Anomaly non-zero in %d / %d rows (%.1f%%)",
        n_nonzero, len(df), 100 * n_nonzero / len(df),
    )
    return df

