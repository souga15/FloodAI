"""
Terrain-derived covariates from SRTM DEM: slope, Topographic Wetness Index
(TWI), and Height Above Nearest Drainage (HAND).

HONESTY NOTE ON HAND-DERIVED DEPTH (read before using in a manuscript):
HAND itself — height above the nearest drainage cell along the flow network
— is a well-established, purely geometric quantity computed directly from
the DEM. This module computes HAND correctly and deterministically.

Converting HAND into an estimated *flood depth in meters* requires a
calibration step: relating HAND values to observed inundation depths during
known flood events. That calibration requires reference data this project
has not yet acquired (e.g. Sentinel-1 SAR-derived flood extents, or gauge
records of inundation depth at known locations/times). Until that reference
data is acquired and the calibration is fitted and validated against held-out
events, `estimate_depth_from_hand()` below returns a depth proxy under an
explicitly stated literature-sourced assumption, tagged as
`calibration_status="uncalibrated_literature_default"` in its output. Any
manuscript figure or table using this output MUST carry that caveat — do not
let an uncalibrated estimate read as a validated prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("floodai.gis.terrain")


def compute_slope_degrees(dem: np.ndarray, cellsize_m: float) -> np.ndarray:
    """Compute slope (degrees) from a DEM array using a simple finite-difference
    gradient. cellsize_m is the ground resolution of one DEM pixel."""
    gy, gx = np.gradient(dem, cellsize_m)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.degrees(slope_rad)


def compute_twi(
    dem: np.ndarray, flow_accumulation: np.ndarray, slope_degrees: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Topographic Wetness Index: TWI = ln( a / tan(beta) )
    where `a` is specific catchment area (flow accumulation x cell area) and
    `beta` is local slope. Standard formulation (Beven & Kirkby, 1979).

    `flow_accumulation` must be precomputed (e.g. via richdem or pysheds —
    this function does not compute flow accumulation itself; that is a
    separate, more involved step requiring D8/D-infinity flow routing).
    """
    slope_rad = np.radians(np.clip(slope_degrees, 0.01, 90))  # avoid tan(0) singularity
    twi = np.log((flow_accumulation + eps) / (np.tan(slope_rad) + eps))
    return twi


def estimate_depth_from_hand(
    hand_values: np.ndarray,
    method: str = "uncalibrated_literature_default",
) -> dict[str, np.ndarray | str]:
    """
    Produce a flood-depth proxy from HAND values.

    method="uncalibrated_literature_default" applies a simple inverse
    relationship (lower HAND -> higher expected depth, capped at a literature
    plausible maximum) purely as a *relative ranking* of inundation
    susceptibility. It is NOT a calibrated depth-in-meters prediction.

    Returns a dict, not a bare array, specifically so the calibration_status
    string travels with the data through the pipeline and can't be silently
    dropped before it reaches a figure or table.
    """
    if method != "uncalibrated_literature_default":
        raise NotImplementedError(
            f"Calibrated HAND-depth method '{method}' is not yet implemented. "
            "This requires reference inundation data (e.g. Sentinel-1 flood "
            "extents) that has not been acquired for this project. "
            "Acquire and validate that data before adding a calibrated method here."
        )

    hand_clipped = np.clip(hand_values, 0, None)
    # Literature-informed plausible maximum local inundation depth (m) used
    # only to bound the proxy to a physically sane range; this is a ceiling,
    # not a fitted parameter.
    plausible_max_depth_m = 5.0
    depth_proxy = plausible_max_depth_m * np.exp(-hand_clipped / 2.0)

    logger.warning(
        "estimate_depth_from_hand: returning UNCALIBRATED depth proxy. "
        "Use only as a relative susceptibility ranking, never as an absolute "
        "depth claim, until calibrated against reference inundation data."
    )
    return {
        "depth_proxy_m": depth_proxy,
        "calibration_status": "uncalibrated_literature_default",
        "caveat": (
            "This is a HAND-based relative susceptibility proxy, not a "
            "calibrated depth prediction. No reference inundation data has "
            "been used to fit this relationship."
        ),
    }
