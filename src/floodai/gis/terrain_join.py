"""
Terrain feature join: adds Elevation_m, TWI, Curve_Number, CN_Runoff_Q
and interaction columns to a points DataFrame.

Data sources (all open, no API key required):
  - Elevation: Open-Elevation API (https://api.open-elevation.com)
    which serves SRTM 30m data. Batched to avoid rate limits.
  - TWI proxy: computed from elevation neighbourhood gradient (slope proxy).
    Full TWI requires flow accumulation grids; this module uses a
    documented slope-only proxy (logged as such) until SRTM tiles are
    available locally for pysheds-based flow routing.
  - Curve Number: SCS-CN lookup table by basin, based on dominant land
    cover and hydrologic soil group documented in literature for each basin.
    Reference: USDA-NRCS TR-55 (1986), Table 2-2.

CN values per basin (literature-sourced, not fitted from data):
  ganga_bihar:       CN=75 (mixed agriculture + moderate-density settlements,
                             HSG-C soils dominant in Indo-Gangetic Plain)
  brahmaputra_assam: CN=72 (forest + shifting cultivation mix, HSG-B/C)
  mahanadi_odisha:   CN=78 (rain-fed agriculture + red laterite soils HSG-C/D)

These are fixed per basin (not per point) because sub-district-level
soil/land-cover maps are not yet integrated. A per-point CN from a
spatial join against HWSD or ISRIC would be the preferred upgrade path.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("floodai.gis.terrain_join")

# ---------------------------------------------------------------------------
# Basin-level SCS Curve Number lookup (literature-sourced, see module doc)
# ---------------------------------------------------------------------------
BASIN_CURVE_NUMBERS: dict[str, int] = {
    "ganga_bihar": 75,
    "brahmaputra_assam": 72,
    "mahanadi_odisha": 78,
}

# ---------------------------------------------------------------------------
# Open-Elevation API config
# ---------------------------------------------------------------------------
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
BATCH_SIZE = 100        # max locations per request (API limit)
RETRY_DELAY_S = 2.0     # seconds between retries on failure
MAX_RETRIES = 3


def _fetch_elevations_batch(latlons: list[tuple[float, float]]) -> list[float | None]:
    """Fetch elevations for a batch of (lat, lon) pairs from Open-Elevation."""
    payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in latlons]}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OPEN_ELEVATION_URL, json=payload, timeout=30)
            resp.raise_for_status()
            results = resp.json()["results"]
            return [r["elevation"] for r in results]
        except Exception as exc:
            logger.warning("Open-Elevation batch attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_S * (attempt + 1))
    return [None] * len(latlons)


def fetch_elevations(points_df: pd.DataFrame) -> pd.Series:
    """
    Fetch SRTM elevation (m) for every row in points_df via Open-Elevation API.
    Returns a pd.Series of floats indexed like points_df.
    Rows where the API fails get NaN (logged with count).
    """
    latlons = list(zip(points_df["lat"], points_df["lon"]))
    elevations: list[float | None] = []

    n_batches = (len(latlons) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("Fetching elevations for %d points in %d batches via Open-Elevation...",
                len(latlons), n_batches)

    for i in range(0, len(latlons), BATCH_SIZE):
        batch = latlons[i: i + BATCH_SIZE]
        elevs = _fetch_elevations_batch(batch)
        elevations.extend(elevs)
        if i + BATCH_SIZE < len(latlons):
            time.sleep(0.5)  # polite rate-limiting

    series = pd.Series(elevations, index=points_df.index, dtype=float)
    n_failed = series.isna().sum()
    if n_failed:
        logger.warning("%d/%d elevation lookups failed (API error). These rows will have NaN Elevation_m.",
                       n_failed, len(series))
    else:
        logger.info("All %d elevation lookups succeeded. Range: %.0f–%.0f m",
                    len(series), series.min(), series.max())
    return series


def compute_twi_proxy(elevation_series: pd.Series, points_df: pd.DataFrame) -> pd.Series:
    """
    TWI proxy using slope estimated from nearest-neighbour elevation differences
    within the same basin. Indexed by points_df.index (integer positions).
    """
    # Work entirely on a reset-index copy to avoid label/position mismatches
    pts = points_df.reset_index(drop=True).copy()
    elevs = elevation_series.reset_index(drop=True)

    twi_proxy = pd.Series(np.nan, index=pts.index, dtype=float)
    for basin in pts["basin_key"].unique():
        mask = pts["basin_key"] == basin          # boolean, same integer index as pts
        basin_elevs = elevs[mask].dropna()
        if len(basin_elevs) < 2:
            continue
        elev_std = basin_elevs.std()
        lat_mean = pts.loc[mask, "lat"].mean()
        deg_to_m = 111_000 * np.cos(np.radians(lat_mean))
        lat_span = pts.loc[mask, "lat"].max() - pts.loc[mask, "lat"].min()
        lon_span = pts.loc[mask, "lon"].max() - pts.loc[mask, "lon"].min()
        spacing_m = max(1.0, np.sqrt((lat_span * 111_000)**2 + (lon_span * deg_to_m)**2) / max(1, len(basin_elevs)))
        slope_proxy = max(elev_std / spacing_m, 1e-4)
        twi_val = float(np.log(1.0 / slope_proxy))
        twi_proxy[mask] = twi_val

    logger.warning(
        "TWI values are a basin-level slope proxy (elevation std / spacing), NOT "
        "full Beven-Kirkby TWI. Label as 'TWI_proxy' in manuscript tables until "
        "pysheds-based flow routing is integrated."
    )
    # Re-index back to original points_df index
    twi_proxy.index = points_df.index
    return twi_proxy


def compute_cn_runoff(rainfall_mm: pd.Series, curve_number: pd.Series) -> pd.Series:
    """
    SCS-CN direct runoff equation: Q = (P - 0.2S)^2 / (P + 0.8S)  if P > 0.2S else 0
    where S = (25400/CN) - 254  [mm]
    Reference: USDA-NRCS TR-55 (1986).
    """
    S = (25400.0 / curve_number.clip(lower=1)) - 254.0
    threshold = 0.2 * S
    excess = (rainfall_mm - threshold).clip(lower=0)
    denom = (rainfall_mm + 0.8 * S).clip(lower=1e-6)
    Q = (excess ** 2) / denom
    return Q


def add_terrain_features(
    df: pd.DataFrame,
    points_df: pd.DataFrame,
    elevation_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Join terrain features into the main feature DataFrame `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Main daily feature DataFrame with columns [point_id, basin_key, Rainfall_mm, ...].
    points_df : pd.DataFrame
        One row per spatial point, with [point_id, basin_key, lat, lon].
        If it already has an 'Elevation_m' column, that is used directly.
        Otherwise, elevations are fetched from Open-Elevation API.
    elevation_col : str, optional
        If points_df already has elevation under a different column name, pass it here.

    Returns
    -------
    pd.DataFrame
        df with new columns: Elevation_m, Curve_Number, TWI, CN_Runoff_Q,
        Elevation_Rain_Ratio, Elevation_Rain30_Ratio, Low_Elev_Heavy_Rain.
    """
    df = df.copy()

    # ---- Step 1: Elevation ------------------------------------------------
    if "Elevation_m" in points_df.columns:
        elev_map = points_df.set_index("point_id")["Elevation_m"]
        logger.info("Using pre-existing Elevation_m column from points_df.")
    elif elevation_col and elevation_col in points_df.columns:
        elev_map = points_df.set_index("point_id")[elevation_col].rename("Elevation_m")
        logger.info("Using elevation from column '%s'.", elevation_col)
    else:
        logger.info("No elevation column found in points_df — fetching from Open-Elevation API.")
        elev_series = fetch_elevations(points_df)
        points_df = points_df.copy()
        points_df["Elevation_m"] = elev_series.values
        elev_map = points_df.set_index("point_id")["Elevation_m"]

    df["Elevation_m"] = df["point_id"].map(elev_map)

    # ---- Step 2: Curve Number (basin-level lookup) ------------------------
    cn_map = {pid: BASIN_CURVE_NUMBERS.get(bk, 75)
              for pid, bk in zip(points_df["point_id"], points_df["basin_key"])}
    df["Curve_Number"] = df["point_id"].map(cn_map).astype(float)

    # ---- Step 3: TWI proxy -----------------------------------------------
    twi_map = compute_twi_proxy(elev_map.rename("Elevation_m"), points_df)
    # twi_map is indexed by points_df index; remap via point_id
    pid_to_twi = dict(zip(points_df["point_id"], twi_map.values))
    df["TWI"] = df["point_id"].map(pid_to_twi)

    # ---- Step 4: SCS-CN Runoff Q -----------------------------------------
    rain_col = "Rainfall_7Day_mm" if "Rainfall_7Day_mm" in df.columns else "Rainfall_mm"
    df["CN_Runoff_Q"] = compute_cn_runoff(df[rain_col].fillna(0), df["Curve_Number"])

    # ---- Step 5: Interaction features ------------------------------------
    elev_clipped = df["Elevation_m"].clip(lower=0.1).fillna(df["Elevation_m"].median())
    df["Elevation_Rain_Ratio"] = df["Rainfall_mm"].fillna(0) / elev_clipped
    df["Elevation_Rain30_Ratio"] = df.get("Rainfall_30Day_mm", df["Rainfall_mm"]).fillna(0) / elev_clipped
    df["Low_Elev_Heavy_Rain"] = (
        (df["Elevation_m"].fillna(999) < 100).astype(float) *
        (df["Rainfall_mm"].fillna(0) > 50).astype(float)
    )

    n_terrain_cols = sum(c in df.columns for c in ["Elevation_m", "Curve_Number", "TWI", "CN_Runoff_Q"])
    logger.info(
        "add_terrain_features: added %d terrain columns + 3 interaction columns. "
        "Elevation range: %.0f–%.0f m. CN range: %d–%d.",
        n_terrain_cols,
        df["Elevation_m"].min(), df["Elevation_m"].max(),
        int(df["Curve_Number"].min()), int(df["Curve_Number"].max()),
    )
    return df
