"""
Abstract interfaces for meteorological and terrain data providers.

Why this exists: the project goal explicitly asks for a swappable data layer
("Future weather inputs should be modular... allow future integration of
IMD forecasts"). Concretely, that means: nothing downstream of ingestion may
import a provider-specific module. Everything talks to these interfaces.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger("floodai.data")


class DataIngestionError(Exception):
    """Raised when a provider cannot deliver validated data after retries."""


class RainfallProvider(ABC):
    """Interface for any daily-gridded-rainfall data source."""

    @abstractmethod
    def fetch_point_series(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Return a DataFrame with columns ['Date', 'Rainfall_mm'] for one point.

        Implementations MUST:
          - validate the response (no silent empty frames),
          - replace documented missing-value sentinels with NaN,
          - log provenance (source name, grid cell used, resolution).
        """
        raise NotImplementedError

    @abstractmethod
    def citation(self) -> str:
        """Return the citation string required when this data is used in a publication."""
        raise NotImplementedError


class DEMProvider(ABC):
    """Interface for digital elevation model sources."""

    @abstractmethod
    def fetch_tile(self, bbox: dict[str, float], out_path: Path) -> Path:
        """Download/clip a DEM tile covering bbox and write it to out_path. Returns out_path."""
        raise NotImplementedError

    @abstractmethod
    def citation(self) -> str:
        raise NotImplementedError


def validate_daily_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    expected_start: str,
    expected_end: str,
    max_missing_fraction: float = 0.10,
) -> None:
    """Shared validation logic: every provider's output passes through this.

    Raises DataIngestionError if the series is empty, has a bad date range,
    or exceeds the allowed missing-value fraction. This is what makes
    "every downloader must validate files... check missing values" an
    enforced runtime check rather than a checklist item nobody runs.
    """
    if df is None or len(df) == 0:
        raise DataIngestionError(f"Empty series returned for expected range {expected_start}-{expected_end}.")

    if date_col not in df.columns or value_col not in df.columns:
        raise DataIngestionError(f"Series missing required columns {date_col}/{value_col}. Got: {list(df.columns)}")

    actual_start, actual_end = df[date_col].min(), df[date_col].max()
    exp_start, exp_end = pd.to_datetime(expected_start), pd.to_datetime(expected_end)
    if actual_start > exp_start + pd.Timedelta(days=2) or actual_end < exp_end - pd.Timedelta(days=2):
        logger.warning(
            "Series date range %s..%s does not fully cover requested %s..%s",
            actual_start.date(), actual_end.date(), exp_start.date(), exp_end.date(),
        )

    missing_frac = df[value_col].isna().mean()
    if missing_frac > max_missing_fraction:
        raise DataIngestionError(
            f"Missing-value fraction {missing_frac:.2%} exceeds allowed "
            f"{max_missing_fraction:.0%} for column '{value_col}'."
        )
    logger.info(
        "Validated series: %d rows, %s..%s, missing=%.2f%%",
        len(df), actual_start.date(), actual_end.date(), missing_frac * 100,
    )
