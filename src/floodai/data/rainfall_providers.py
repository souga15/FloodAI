"""
IMD 0.25-degree gridded daily rainfall provider.

Data: Pai et al. (2014), "Development of a new high spatial resolution
(0.25 x 0.25) long period (1901-2010) daily gridded rainfall dataset over
India", MAUSAM 65(1):1-18. Distributed by IMD Pune (imdpune.gov.in).

IMPORTANT — read before trusting any number derived from this module:
This provider was written against the documented `imdlib` package API and
IMD's published file format. It has NOT been executed against the live IMD
server from this development environment, because imdpune.gov.in is outside
the sandboxed network this code was authored in. The first time you run
`collect_basin_rainfall()` in your own Colab runtime, you MUST inspect:
    1. that the returned date range matches what you requested,
    2. that SMAP/rainfall values are in a physically plausible range
       (0-500mm/day for India; >1000mm/day is almost certainly a sentinel
       value that slipped through),
    3. the logged missing-value fraction per grid cell.
Do not proceed to feature engineering until you have done this check once.
A `validate_first_run.py` script is provided in notebooks/ for exactly this.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from floodai.data.base import DataIngestionError, RainfallProvider, validate_daily_series

logger = logging.getLogger("floodai.data.imd")

IMD_GRID_RESOLUTION_DEG = 0.25
IMD_LAT_ORIGIN = 6.5
IMD_LON_ORIGIN = 66.5
IMD_MISSING_SENTINEL = -999.0


class IMDGriddedRainfallProvider(RainfallProvider):
    """
    Wraps `imdlib` (https://imdlib.readthedocs.io) to fetch IMD 0.25-degree
    gridded daily rainfall and extract a point time series via nearest-grid-cell
    lookup. Caches downloaded yearly files to `cache_dir` so re-runs don't
    re-download (IMD's server has no documented rate limit, but yearly grid
    files are large; caching is good practice regardless).
    """

    def __init__(self, cache_dir: str | Path = "data/raw/imd_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._yearly_cache: dict[int, "object"] = {}  # year -> xarray.Dataset

    def citation(self) -> str:
        return (
            "Pai, D.S., Sridhar, L., Rajeevan, M., Sreejith, O.P., Satbhai, N.S. "
            "and Mukhopadhyay, B. (2014). Development of a new high spatial "
            "resolution (0.25 x 0.25) long period (1901-2010) daily gridded "
            "rainfall data set over India and its comparison with existing "
            "data sets over the region. MAUSAM, 65(1), 1-18."
        )

    def _load_year(self, year: int):
        """Download (or load from cache) one year of gridded data via imdlib."""
        if year in self._yearly_cache:
            return self._yearly_cache[year]

        try:
            import imdlib
        except ImportError as e:
            raise DataIngestionError(
                "imdlib is not installed. Run: pip install imdlib --break-system-packages"
            ) from e

        # Check if local file exists in the directory structure
        local_file = self.cache_dir / "rain" / f"{year}.grd"
        if local_file.exists():
            logger.info("Loading local IMD gridded rainfall for year %d from %s...", year, local_file)
            try:
                data = imdlib.open_data(
                    "rain", year, year, fn_format="yearwise", file_dir=str(self.cache_dir)
                )
                ds = data.get_xarray()
                self._yearly_cache[year] = ds
                return ds
            except Exception as e:
                logger.warning("Failed to open local IMD file %s: %s. Falling back to downloading.", local_file, e)

        logger.info("Fetching IMD gridded rainfall for year %d (cache_dir=%s)...", year, self.cache_dir)
        try:
            data = imdlib.get_data(
                "rain", year, year, fn_format="yearwise", file_dir=str(self.cache_dir)
            )
            ds = data.get_xarray()
            time.sleep(1.0)  # be polite to the source server after a download
        except Exception as e:
            raise DataIngestionError(
                f"imdlib failed to fetch/parse IMD rainfall for {year}: {e}"
            ) from e

        self._yearly_cache[year] = ds
        return ds

    def fetch_point_series(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> pd.DataFrame:
        start = pd.to_datetime(start_date, format="%Y%m%d")
        end = pd.to_datetime(end_date, format="%Y%m%d")
        years = list(range(start.year, end.year + 1))

        frames = []
        for year in years:
            ds = self._load_year(year)
            # Nearest-neighbour lookup on the 0.25-degree grid.
            point = ds["rain"].sel(lat=lat, lon=lon, method="nearest")
            df_year = point.to_dataframe().reset_index()[["time", "rain"]]
            df_year.columns = ["Date", "Rainfall_mm"]
            frames.append(df_year)

        df = pd.concat(frames, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[(df["Date"] >= start) & (df["Date"] <= end)].reset_index(drop=True)
        df["Rainfall_mm"] = df["Rainfall_mm"].replace(IMD_MISSING_SENTINEL, np.nan)

        validate_daily_series(
            df, date_col="Date", value_col="Rainfall_mm",
            expected_start=start_date, expected_end=end_date,
        )
        return df


class NASAPowerFallbackProvider(RainfallProvider):
    """
    Documented fallback/comparison provider, kept for the explicit sensitivity
    comparison the project calls for (IMD ground-truth vs satellite/model
    reanalysis). This is the same NASA POWER endpoint used in the prior WPT
    notebook — that ingestion logic was sound and is reused, not rewritten.
    Use this provider for the rainfall-bias sensitivity analysis, not as the
    primary rainfall source for headline results (config.yaml fixes IMD as
    primary; this class exists for the comparison study only).
    """

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def citation(self) -> str:
        return "NASA POWER Project, https://power.larc.nasa.gov (GEOS-5 reanalysis)."

    def fetch_point_series(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> pd.DataFrame:
        import requests

        url = (
            f"{self.BASE_URL}?parameters=PRECTOTCORR&community=AG&"
            f"longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        series = payload["properties"]["parameter"]["PRECTOTCORR"]
        df = pd.DataFrame({
            "Date": pd.to_datetime(list(series.keys()), format="%Y%m%d"),
            "Rainfall_mm": list(series.values()),
        })
        df["Rainfall_mm"] = df["Rainfall_mm"].replace(-999, np.nan)
        validate_daily_series(
            df, date_col="Date", value_col="Rainfall_mm",
            expected_start=start_date, expected_end=end_date,
        )
        return df


def get_rainfall_provider(provider_name: str, **kwargs) -> RainfallProvider:
    """Factory: dispatch on config['data_sources']['rainfall']['provider']."""
    registry = {
        "imd_gridded_0p25": IMDGriddedRainfallProvider,
        "nasa_power": NASAPowerFallbackProvider,
    }
    if provider_name not in registry:
        raise DataIngestionError(
            f"Unknown rainfall provider '{provider_name}'. Available: {list(registry)}"
        )
    return registry[provider_name](**kwargs)
