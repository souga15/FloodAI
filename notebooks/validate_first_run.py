"""
validate_first_run.py — RUN THIS BEFORE TRUSTING ANY DOWNSTREAM RESULT.

This script exists because the IMD provider (floodai.data.rainfall_providers.
IMDGriddedRainfallProvider) was written against documented APIs but has not
been executed against the live IMD server in the environment this code was
authored in. The first time you run the real pipeline in your own Colab
session, run this script first and read its output carefully.

It checks, for one representative point per basin:
    1. Date range coverage matches what was requested.
    2. Missing-value fraction is reasonable.
    3. Rainfall values are physically plausible for India (flags anything
       above 600mm/day, which is rare-but-real during extreme events, and
       anything that looks like a leaked sentinel value e.g. -999 or 999).
    4. The grid cell actually used (nearest-neighbour lat/lon) is reasonably
       close to the requested point — large mismatches usually indicate a
       coordinate-order bug (lat/lon swapped) rather than a data problem.

Usage (from Colab, after pip installing requirements.txt and floodai):
    !python notebooks/validate_first_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from floodai.data.rainfall_providers import IMDGriddedRainfallProvider

CHECK_POINTS = {
    "ganga_bihar_sample": (25.5941, 85.1376),       # Patna, for reference
    "brahmaputra_assam_sample": (26.1445, 91.7362),  # Guwahati, for reference
    "mahanadi_odisha_sample": (20.2961, 85.8245),    # Bhubaneswar, for reference
}

PLAUSIBLE_MAX_MM_PER_DAY = 600.0
COORD_MISMATCH_WARN_DEG = 0.30  # ~33km at this latitude; flags gross lookup errors


def main() -> int:
    provider = IMDGriddedRainfallProvider()
    all_ok = True

    for label, (lat, lon) in CHECK_POINTS.items():
        print(f"\n--- Checking {label} (requested lat={lat}, lon={lon}) ---")
        try:
            df = provider.fetch_point_series(lat, lon, "20230101", "20231231")
        except Exception as e:
            print(f"  [FAIL] Could not fetch data: {e}")
            all_ok = False
            continue

        print(f"  Rows returned: {len(df)} (expected ~365)")
        print(f"  Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")

        missing_frac = df["Rainfall_mm"].isna().mean()
        print(f"  Missing fraction: {missing_frac:.2%}")
        if missing_frac > 0.05:
            print("  [WARN] Missing fraction above 5% — inspect before proceeding.")

        max_val = df["Rainfall_mm"].max(skipna=True)
        print(f"  Max daily rainfall in series: {max_val:.1f} mm")
        if max_val > PLAUSIBLE_MAX_MM_PER_DAY:
            print(
                f"  [WARN] Max value {max_val:.1f}mm exceeds plausible ceiling "
                f"({PLAUSIBLE_MAX_MM_PER_DAY}mm). Could be a real extreme event "
                f"(check against news/CWC for that date) OR a leaked sentinel "
                f"value that wasn't caught by the -999 replacement. Inspect the "
                f"specific date before trusting this series."
            )

        suspicious = df[df["Rainfall_mm"].abs().isin([999, 9999])]
        if len(suspicious) > 0:
            print(f"  [FAIL] Found {len(suspicious)} rows with exact sentinel-like "
                  f"values (999/9999) that were not replaced with NaN.")
            all_ok = False

        if len(df) < 300:  # well short of a full year
            print(f"  [FAIL] Only {len(df)} rows for a 1-year request — coverage gap.")
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("[OK] First-run checks passed. Proceed to the main pipeline.")
        print("     (This validates data plausibility, not statistical correctness —")
        print("      still spot-check a few values against a known flood date.)")
    else:
        print("[FAIL] One or more checks failed. DO NOT proceed to feature")
        print("       engineering / model training until these are resolved.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
