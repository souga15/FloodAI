"""
Real terrain feature join: per-point Curve Number from ISRIC SoilGrids +
real TWI from pysheds flow routing on an Open-Elevation API grid DEM.

Data sources (all open access, NO API key required):
  - Soil data: ISRIC SoilGrids REST API v2 (https://rest.isric.org)
    clay + sand content at 0-30cm depth -> Hydrologic Soil Group -> CN
  - Elevation/TWI: Open-Elevation API (SRTM data) queried on a dense
    0.05 degree grid per basin -> pysheds D8 flow routing -> real TWI
    No system tools (gdal/make) needed, unlike the `elevation` package.

CN lookup table: USDA-NRCS TR-55 (1986), Table 2-2.
HSG classification: USDA NRCS, Part 630 Hydrology, Chapter 7.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("floodai.gis.terrain_real")

# ---------------------------------------------------------------------------
# Hydrologic Soil Group from clay + sand content (USDA NRCS Part 630 Ch.7)
# ---------------------------------------------------------------------------
def classify_hsg(clay_pct: float, sand_pct: float) -> str:
    if clay_pct > 40:
        return "D"
    elif clay_pct > 25:
        return "C"
    elif sand_pct > 70 and clay_pct < 10:
        return "A"
    else:
        return "B"


# CN lookup: land cover x HSG (USDA TR-55 Table 2-2)
CN_TABLE: dict[str, dict[str, int]] = {
    "cropland":  {"A": 67, "B": 78, "C": 85, "D": 89},
    "grassland": {"A": 30, "B": 58, "C": 71, "D": 78},
    "forest":    {"A": 36, "B": 60, "C": 73, "D": 79},
    "urban":     {"A": 77, "B": 85, "C": 90, "D": 92},
    "bare":      {"A": 72, "B": 82, "C": 87, "D": 89},
    "water":     {"A": 98, "B": 98, "C": 98, "D": 98},
    "wetland":   {"A": 78, "B": 78, "C": 78, "D": 78},
}

ESA_TO_CN_CLASS: dict[int, str] = {
    10: "forest", 20: "forest", 30: "grassland", 40: "cropland",
    50: "urban",  60: "bare",   70: "bare",       80: "water",
    90: "wetland", 95: "wetland", 100: "bare",
}

# ---------------------------------------------------------------------------
# ISRIC SoilGrids API — per-point soil texture -> HSG -> CN
# ---------------------------------------------------------------------------
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"


def fetch_soil_properties(lat: float, lon: float) -> dict[str, float] | None:
    """Fetch clay and sand (%) at 0-30cm from ISRIC SoilGrids."""
    params = {
        "lon": lon, "lat": lat,
        "property": ["clay", "sand"],
        "depth": ["0-5cm", "5-15cm", "15-30cm"],
        "value": ["mean"],
    }
    try:
        resp = requests.get(SOILGRIDS_URL, params=params, timeout=20)
        resp.raise_for_status()
        result = {}
        for layer in resp.json().get("properties", {}).get("layers", []):
            name = layer["name"]
            vals = [d["values"].get("mean") for d in layer.get("depths", [])
                    if d["values"].get("mean") is not None]
            if vals:
                result[name] = float(np.mean(vals)) / 10.0  # g/kg -> %
        return result or None
    except Exception as exc:
        logger.debug("SoilGrids failed (%.4f, %.4f): %s", lat, lon, exc)
        return None


def get_per_point_cn(points_df: pd.DataFrame, lc_col: Optional[str] = None) -> pd.Series:
    """Per-point CN from ISRIC SoilGrids. Falls back to basin defaults."""
    basin_defaults = {"ganga_bihar": "B", "brahmaputra_assam": "B", "mahanadi_odisha": "C"}
    cn_series = pd.Series(np.nan, index=points_df.index, dtype=float)
    n_api, n_fallback = 0, 0

    logger.info("Fetching per-point CN from ISRIC SoilGrids (%d points)...", len(points_df))
    for idx, row in points_df.iterrows():
        lc_name = ESA_TO_CN_CLASS.get(int(row[lc_col]), "cropland") if lc_col and lc_col in points_df.columns else "cropland"
        soil = fetch_soil_properties(row["lat"], row["lon"])
        time.sleep(0.15)
        if soil and "clay" in soil and "sand" in soil:
            hsg = classify_hsg(soil["clay"], soil["sand"])
            n_api += 1
        else:
            hsg = basin_defaults.get(str(row.get("basin_key", "")), "B")
            n_fallback += 1
        cn_series[idx] = CN_TABLE.get(lc_name, CN_TABLE["cropland"])[hsg]

    logger.info("CN complete: %d API, %d defaults. Range: %.0f-%.0f",
                n_api, n_fallback, cn_series.min(), cn_series.max())
    return cn_series


# ---------------------------------------------------------------------------
# TWI from Open-Elevation grid + pysheds (no API key, no system tools)
# ---------------------------------------------------------------------------
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
BATCH_SIZE = 100


def _fetch_elevations_batch(latlons: list[tuple[float, float]]) -> list[float]:
    payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in latlons]}
    for attempt in range(3):
        try:
            resp = requests.post(OPEN_ELEVATION_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return [r["elevation"] for r in resp.json()["results"]]
        except Exception as exc:
            logger.warning("Open-Elevation batch attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2.0 * (attempt + 1))
    return [0.0] * len(latlons)


def _build_basin_dem(lat_min: float, lat_max: float,
                     lon_min: float, lon_max: float,
                     resolution_deg: float = 0.05) -> tuple[np.ndarray, dict]:
    """
    Query Open-Elevation API on a regular grid to build a basin DEM array.
    Resolution 0.05 deg (~5km). Returns (dem_array_northup, grid_info).
    """
    lats = np.arange(lat_min, lat_max + resolution_deg / 2, resolution_deg)
    lons = np.arange(lon_min, lon_max + resolution_deg / 2, resolution_deg)
    grid_latlons = [(lat, lon) for lat in lats for lon in lons]

    logger.info("Querying Open-Elevation for basin DEM: %d x %d = %d points...",
                len(lats), len(lons), len(grid_latlons))

    elevations = []
    for i in range(0, len(grid_latlons), BATCH_SIZE):
        batch = grid_latlons[i: i + BATCH_SIZE]
        elevations.extend(_fetch_elevations_batch(batch))
        if i + BATCH_SIZE < len(grid_latlons):
            time.sleep(0.3)

    elev_arr = np.array(elevations, dtype=float).reshape(len(lats), len(lons))
    elev_arr = np.flipud(elev_arr)  # row 0 = northernmost

    return elev_arr, {"lats": lats, "lons": lons, "res": resolution_deg,
                      "lat_min": lat_min, "lat_max": lat_max,
                      "lon_min": lon_min, "lon_max": lon_max}


def _sample_grid(grid_arr: np.ndarray, info: dict, lat: float, lon: float) -> float:
    """Nearest-neighbour sample of a grid array at a lat/lon."""
    lats_flip = np.flipud(info["lats"])
    r = int(np.argmin(np.abs(lats_flip - lat)))
    c = int(np.argmin(np.abs(info["lons"] - lon)))
    r = int(np.clip(r, 0, grid_arr.shape[0] - 1))
    c = int(np.clip(c, 0, grid_arr.shape[1] - 1))
    return float(grid_arr[r, c])


def compute_real_twi(points_df: pd.DataFrame, dem_cache_dir: str = "/tmp/dem") -> pd.Series:
    """
    Real TWI per point using pysheds on an Open-Elevation API grid DEM.
    No API key, no GDAL/make system tools required.
    Resolution: 0.05 deg (~5km) — real spatial variation within each basin.
    Requires: pip install pysheds affine
    """
    try:
        from pysheds.grid import Grid
        from pysheds.view import Raster, ViewFinder
        import affine as affine_lib
    except ImportError as exc:
        raise RuntimeError("Run: !pip install pysheds affine") from exc

    twi_series = pd.Series(np.nan, index=points_df.index, dtype=float)
    res = 0.05

    for basin in points_df["basin_key"].unique():
        mask = points_df["basin_key"] == basin
        basin_pts = points_df[mask].copy().reset_index()  # keep orig index

        lat_min = basin_pts["lat"].min() - 0.15
        lat_max = basin_pts["lat"].max() + 0.15
        lon_min = basin_pts["lon"].min() - 0.15
        lon_max = basin_pts["lon"].max() + 0.15

        elev_arr, info = _build_basin_dem(lat_min, lat_max, lon_min, lon_max, res)
        cellsize_m = res * 111_000

        # Build pysheds Raster
        aff = affine_lib.Affine(res, 0, lon_min, 0, -res, lat_max)
        vf = ViewFinder(affine=aff, shape=elev_arr.shape, nodata=-9999.0)
        dem_raster = Raster(elev_arr.copy(), viewfinder=vf)
        grid = Grid(viewfinder=vf)

        # Condition DEM
        try:
            conditioned = grid.resolve_flats(
                grid.fill_depressions(grid.fill_pits(dem_raster))
            )
        except Exception as exc:
            logger.warning("DEM conditioning failed for '%s': %s", basin, exc)
            conditioned = dem_raster

        # Flow routing
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        try:
            fdir = grid.flowdir(conditioned, dirmap=dirmap)
            acc_arr = np.array(grid.accumulation(fdir, dirmap=dirmap), dtype=float)
        except Exception as exc:
            logger.warning("Flow routing failed for '%s': %s — uniform acc", basin, exc)
            acc_arr = np.ones_like(elev_arr, dtype=float)

        # Slope and TWI
        gy, gx = np.gradient(np.array(conditioned, dtype=float), cellsize_m)
        slope_rad = np.clip(np.arctan(np.sqrt(gx**2 + gy**2)), 0.001, np.pi / 2)
        twi_grid = np.clip(
            np.log((acc_arr * cellsize_m**2 + 1.0) / np.tan(slope_rad)),
            0, 30
        )

        # Sample at each point
        for _, row in basin_pts.iterrows():
            orig_idx = row["index"]
            twi_series[orig_idx] = _sample_grid(twi_grid, info, row["lat"], row["lon"])

        logger.info("Basin '%s' TWI: %.2f-%.2f (mean=%.2f)",
                    basin, twi_series[mask].min(),
                    twi_series[mask].max(), twi_series[mask].mean())

    n_nan = twi_series.isna().sum()
    if n_nan:
        twi_series.fillna(twi_series.median(), inplace=True)
        logger.warning("Filled %d NaN TWI with median.", n_nan)
    return twi_series


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def add_real_terrain_features(
    df: pd.DataFrame,
    points_df: pd.DataFrame,
    dem_cache_dir: str = "/content/dem_cache",
) -> pd.DataFrame:
    """
    Full real terrain join (no API key needed):
      1. Elevation: Open-Elevation API (real SRTM values)
      2. CN: ISRIC SoilGrids per-point soil texture -> HSG -> TR-55 CN
      3. TWI: Open-Elevation grid DEM -> pysheds D8 flow routing
      4. Derived: CN_Runoff_Q, interaction features
    """
    from floodai.gis.terrain_join import fetch_elevations, compute_cn_runoff

    df = df.copy()

    # Step 1: Elevation
    if "Elevation_m" not in points_df.columns:
        logger.info("Fetching SRTM elevations via Open-Elevation API...")
        pts = points_df.copy()
        pts["Elevation_m"] = fetch_elevations(points_df).values
        points_df = pts
    elev_map = points_df.set_index("point_id")["Elevation_m"]
    df["Elevation_m"] = df["point_id"].map(elev_map)

    # Step 2: Per-point CN from ISRIC SoilGrids
    logger.info("Step 2: Per-point CN from ISRIC SoilGrids...")
    cn_series = get_per_point_cn(points_df)
    pid_to_cn = dict(zip(points_df["point_id"], cn_series.values))
    df["Curve_Number"] = df["point_id"].map(pid_to_cn).astype(float)

    # Step 3: Real TWI
    logger.info("Step 3: Real TWI via Open-Elevation grid + pysheds...")
    twi_series = compute_real_twi(points_df, dem_cache_dir=dem_cache_dir)
    pid_to_twi = dict(zip(points_df["point_id"], twi_series.values))
    df["TWI"] = df["point_id"].map(pid_to_twi)

    # Step 4: Runoff and interaction features
    rain_col = "Rainfall_7Day_mm" if "Rainfall_7Day_mm" in df.columns else "Rainfall_mm"
    df["CN_Runoff_Q"] = compute_cn_runoff(df[rain_col].fillna(0), df["Curve_Number"])
    elev_c = df["Elevation_m"].clip(lower=0.1).fillna(df["Elevation_m"].median())
    df["Elevation_Rain_Ratio"]   = df["Rainfall_mm"].fillna(0) / elev_c
    df["Elevation_Rain30_Ratio"] = df.get("Rainfall_30Day_mm", df["Rainfall_mm"]).fillna(0) / elev_c
    df["Low_Elev_Heavy_Rain"]    = ((df["Elevation_m"].fillna(999) < 100).astype(float) *
                                    (df["Rainfall_mm"].fillna(0) > 50).astype(float))
    df["CN_Rain_Interaction"]    = df["Curve_Number"] * df[rain_col].fillna(0)
    df["TWI_Rain_Interaction"]   = df["TWI"].fillna(0) * df[rain_col].fillna(0)

    logger.info("Real terrain join complete. Elevation %.0f-%.0f m | CN %.0f-%.0f | TWI %.2f-%.2f",
                df["Elevation_m"].min(), df["Elevation_m"].max(),
                df["Curve_Number"].min(), df["Curve_Number"].max(),
                df["TWI"].min(), df["TWI"].max())
    return df
