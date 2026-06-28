"""
FloodAI: A reproducible, leakage-free flood occurrence prediction framework
for basin-scale hydrological risk assessment.

Scope of this milestone (v0.1):
    - Flood occurrence prediction (binary) across three river basins:
      Ganga (Bihar), Brahmaputra (Assam), Mahanadi (Odisha).
    - Temporal + Leave-One-Basin-Out validation.
    - Held-out-only reporting: no metric in this codebase is ever computed
      on training-inclusive data and reported as a performance estimate.

Explicitly out of scope for v0.1 (tracked as future milestones, not silently
dropped): inundation-depth (HAND) calibration against satellite flood masks,
multi-horizon forecasting, deep learning baselines. These require reference
data (e.g. Sentinel-1 flood extents) that has not yet been acquired/verified.
"""

__version__ = "0.1.0"
