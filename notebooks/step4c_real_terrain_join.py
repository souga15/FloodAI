"""
Step 4c — Real Terrain Join (CN from ISRIC SoilGrids + TWI from pysheds/SRTM)
Paste this as a code cell. Run AFTER Step 4 (feature engineering).
Replaces the proxy terrain join (step4b). Takes ~10-15 minutes.
"""
import sys, importlib
sys.path.insert(0, "/content/FloodAI/src")

# Install required packages (only needed once per Colab session)
import subprocess
subprocess.run(["pip", "install", "-q", "pysheds", "elevation", "rasterio"], check=True)

import floodai.gis.terrain_real as tr
importlib.reload(tr)
from floodai.gis.terrain_real import add_real_terrain_features

print("Starting real terrain join (ISRIC SoilGrids + pysheds SRTM)...")
print("Expected time: 10-15 minutes\n")

df = add_real_terrain_features(df, points_df, dem_cache_dir="/content/dem_cache")

# Additional interaction columns
df["CN_Rain_Interaction"]  = df["Curve_Number"] * df["Rainfall_7Day_mm"].fillna(0)
df["TWI_Rain_Interaction"] = df["TWI"].fillna(0) * df["Rainfall_7Day_mm"].fillna(0)

print("\n=== REAL TERRAIN FEATURES SUMMARY ===")
print(df[["Elevation_m", "Curve_Number", "TWI", "CN_Runoff_Q"]].describe().round(2))
print(f"\nUnique CN values (per-point): {sorted(df['Curve_Number'].unique())}")
print(f"TWI range: {df['TWI'].min():.2f} to {df['TWI'].max():.2f}")
print("\n[OK] Real terrain join complete. Proceed to labelling and model training.")
