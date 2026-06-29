"""
Step 4c — Real Terrain Join (CN from ISRIC SoilGrids + TWI from pysheds/USGS 3DEP)

Paste this as a code cell. Run AFTER Step 4a (feature engineering).
Replaces the proxy terrain join (step4b). Takes ~10-15 minutes.

Uses py3dep (USGS 3DEP, 30m, free, no API key) as primary DEM source.
Falls back automatically to Open-Elevation API if py3dep is unavailable.

Run once per Colab session — terrain is saved to parquet so it does NOT
need to be re-fetched if you only need to rerun model training.
"""
import sys, importlib
from pathlib import Path
sys.path.insert(0, "/content/FloodAI/src")

# Install required packages (only needed once per session)
import subprocess
subprocess.run(["pip", "install", "-q", "pysheds", "affine", "py3dep", "xarray"], check=True)

import floodai.gis.terrain_real as tr
importlib.reload(tr)
from floodai.gis.terrain_real import add_real_terrain_features

TERRAIN_CACHE = Path("/content/floodai_outputs/terrain_cache.parquet")

if TERRAIN_CACHE.exists():
    print(f"[CACHE HIT] Loading terrain from {TERRAIN_CACHE} — skipping API calls.")
    import pandas as pd
    terrain_cols = ["point_id", "Elevation_m", "Curve_Number", "TWI"]
    terrain_cache_df = pd.read_parquet(TERRAIN_CACHE)
    # Merge cached terrain back onto df
    df = df.drop(columns=[c for c in terrain_cols[1:] if c in df.columns], errors="ignore")
    df = df.merge(terrain_cache_df[terrain_cols], on="point_id", how="left")
    # Recompute derived columns that depend on terrain
    from floodai.features.pipeline import add_scs_cn_runoff, add_interaction_features
    df = add_scs_cn_runoff(df)
    df = add_interaction_features(df)
    print("[OK] Terrain restored from cache.")
else:
    print("Starting real terrain join (ISRIC SoilGrids + py3dep USGS 3DEP)...")
    print("Primary DEM: USGS 3DEP 30m via py3dep")
    print("Fallback DEM: Open-Elevation API (SRTM ~5km)")
    print("Expected time: 10-15 minutes\n")
    TERRAIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df = add_real_terrain_features(df, points_df, dem_cache_dir="/content/dem_cache")
    # Save terrain per-point so next session can reload without API calls
    import pandas as pd
    terrain_to_cache = df[["point_id", "Elevation_m", "Curve_Number", "TWI"]].drop_duplicates("point_id")
    terrain_to_cache.to_parquet(TERRAIN_CACHE, index=False)
    print(f"[OK] Terrain cached to {TERRAIN_CACHE} for future sessions.")

# --- Diagnostic: verify real spatial variation ---
print("\n=== REAL TERRAIN FEATURES SUMMARY ===")
print(df[["Elevation_m", "Curve_Number", "TWI", "CN_Runoff_Q"]].describe().round(2))
print(f"\nElevation std (need > 5m for terrain to matter in model): {df['Elevation_m'].std():.1f}m")
print(f"TWI std (need > 1.0 for TWI to carry signal): {df['TWI'].std():.2f}")
print(f"CN range: {df['Curve_Number'].min():.0f} - {df['Curve_Number'].max():.0f}")
print(f"CN_Runoff_Q non-zero rows: {(df['CN_Runoff_Q'] > 0).sum():,} / {len(df):,}")

per_basin = df.groupby("basin_key")["TWI"].agg(["min", "max", "std"]).round(2)
print(f"\nTWI per basin:\n{per_basin}")

if df["Elevation_m"].std() < 5:
    print("\n[WARNING] Elevation std < 5m — terrain features will have low SHAP importance.")
    print("          Consider expanding bbox padding in terrain_real.py or switching to")
    print("          real SRTM 30m tiles (HydroSHEDS) for better within-basin variation.")
else:
    print("\n[OK] Sufficient terrain variation detected — TWI/CN should appear in SHAP.")
