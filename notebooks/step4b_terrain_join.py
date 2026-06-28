"""
Step 4b — Terrain Feature Join
Run this cell AFTER Step 4 (feature engineering) and BEFORE Step 5 (labelling).
This adds Elevation_m, TWI, Curve_Number, CN_Runoff_Q and interaction features.
"""
import importlib
import floodai.gis.terrain_join as tj
importlib.reload(tj)

from floodai.gis.terrain_join import add_terrain_features

print("Fetching SRTM elevations via Open-Elevation API (may take ~60s)...")
df = add_terrain_features(df, points_df)

# Add two more interaction columns from the governance allowlist
df["CN_Rain_Interaction"]  = df["Curve_Number"] * df["Rainfall_7Day_mm"].fillna(0)
df["TWI_Rain_Interaction"] = df["TWI"].fillna(0) * df["Rainfall_7Day_mm"].fillna(0)

terrain_cols = ["Elevation_m", "Curve_Number", "TWI", "CN_Runoff_Q",
                "Elevation_Rain_Ratio", "Elevation_Rain30_Ratio",
                "Low_Elev_Heavy_Rain", "CN_Rain_Interaction", "TWI_Rain_Interaction"]

print("\nTerrain feature sample (first 3 rows):")
print(df[terrain_cols].head(3).to_string())

print(f"\nNaN counts in terrain columns:")
print(df[terrain_cols].isna().sum())
print("\n[OK] Terrain join complete. Proceed to Step 5 (labelling).")
