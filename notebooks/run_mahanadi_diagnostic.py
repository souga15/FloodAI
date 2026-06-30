import sys
from pathlib import Path
sys.path.insert(0, "src")

import logging
import pandas as pd
from floodai.config import load_config
from floodai.gis.points import generate_grid_fallback_points, basin_points_to_dataframe
from floodai.data.rainfall_providers import get_rainfall_provider
from floodai.features.pipeline import add_temporal_features, add_rainfall_window_features

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Load config
cfg = load_config(Path("config/config.yaml"))

print("--- Generating points ---")
all_points = []
for basin_key, basin_cfg in cfg.basins.items():
    pts = generate_grid_fallback_points(
        basin_key=basin_key,
        bbox=basin_cfg.bbox,
        n_points_target=basin_cfg.n_points_target,
        seed=cfg.random_seed,
    )
    all_points.extend(pts)
points_df = basin_points_to_dataframe(all_points)
print(f"Generated {len(points_df)} points across {points_df['basin_key'].nunique()} basins")

print("\n--- Fetching rainfall (Yearly files are downloaded once and cached) ---")
provider = get_rainfall_provider(
    cfg.raw["data_sources"]["rainfall"]["provider"],
)
start_year = cfg.raw["data_sources"]["rainfall"]["start_year"]
end_year = cfg.raw["data_sources"]["rainfall"]["end_year"]

all_series = []
for i, row in points_df.iterrows():
    try:
        df_point = provider.fetch_point_series(
            row["lat"], row["lon"], f"{start_year}0101", f"{end_year}1231"
        )
        df_point["point_id"] = row["point_id"]
        df_point["basin_key"] = row["basin_key"]
        all_series.append(df_point)
    except Exception as e:
        print(f"  [SKIP] Point {row['point_id']} in {row['basin_key']} failed: {e}")
    if (i + 1) % 20 == 0 or (i + 1) == len(points_df):
        print(f"  {i+1}/{len(points_df)} points fetched...")

rainfall_df = pd.concat(all_series, ignore_index=True)
print(f"Rainfall fetch complete: {len(rainfall_df):,} rows.")

print("\n--- Building temporal & rainfall features ---")
df = rainfall_df.merge(points_df[["point_id", "lat", "lon", "basin_key"]], on=["point_id", "basin_key"])
df = df.sort_values(["point_id", "Date"]).reset_index(drop=True)
df = add_temporal_features(df)
df = add_rainfall_window_features(df, group_col="point_id")

print("\n--- Loading flood event labels ---")
flood_events_path = Path(cfg.raw["data_sources"]["flood_events"]["path"])
flood_events_df = pd.read_csv(flood_events_path, parse_dates=["Start", "End"])
# Rename 'Basin' to 'basin_key' to align with label_diagnostics expectations
flood_events_df = flood_events_df.rename(columns={"Basin": "basin_key"})
print(f"Loaded {len(flood_events_df)} flood events.")

# Label floods
def label_floods(df, flood_events_df):
    df = df.copy()
    df['Flood_Occurred'] = 0
    for _, ev in flood_events_df.iterrows():
        mask = (
            (df['basin_key'] == ev['basin_key']) &
            (df['Date'] >= ev['Start']) &
            (df['Date'] <= ev['End'])
        )
        df.loc[mask, 'Flood_Occurred'] = 1
    return df

df = label_floods(df, flood_events_df)

print("\n--- Running label diagnostics ---")
from floodai.evaluation.label_diagnostics import (
    compare_basins_rainfall_coincidence,
    compute_event_rainfall_context,
)

for bk in df['basin_key'].unique():
    print(f"\n" + "="*70)
    print(f"  EVENT-BY-EVENT RAINFALL CONTEXT: {bk.upper()}")
    print("="*70)
    try:
        event_table = compute_event_rainfall_context(
            df, flood_events_df, basin_key=bk, rainfall_col="Rainfall_7Day_mm",
        )
        print(event_table.to_string(index=False))
    except Exception as e:
        print(f"Error running for {bk}: {e}")

print(f"\n" + "="*70)
print("  CROSS-BASIN COMPARISON SUMMARY")
print("="*70)
try:
    summary = compare_basins_rainfall_coincidence(
        df, flood_events_df, basin_keys=list(df['basin_key'].unique()),
        rainfall_col="Rainfall_7Day_mm"
    )
    print(summary.to_string(index=False))
except Exception as e:
    print(f"Error running summary: {e}")
