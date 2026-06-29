"""
Step 2 to 4a: Fetch IMD Data, Label Floods, and Build Initial Features.
This extracts the feature engineering logic that was previously directly embedded in the notebook.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/content/FloodAI/src")
from floodai.data.rainfall_providers import get_rainfall_provider
from floodai.features.pipeline import (
    add_temporal_features, add_rainfall_window_features,
    compute_rainfall_climatology, add_rainfall_anomaly_features,
)
from floodai.config import load_config
import logging

logger = logging.getLogger("floodai.features")
cfg = load_config(Path("config/config.yaml"))

print("--- Step 2: Fetching IMD Rainfall Data ---")
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
        logger.error("Failed to fetch point %s: %s", row["point_id"], e)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(points_df)} points fetched...")

rainfall_df = pd.concat(all_series, ignore_index=True)
rainfall_df.to_parquet(cfg.output_dir / "rainfall_raw.parquet")
print(f"Collected {len(rainfall_df):,} point-days.")

print("\n--- Step 3: Loading Flood Event Labels ---")
flood_events_path = Path(cfg.raw["data_sources"]["flood_events"]["path"])
if not flood_events_path.exists():
    raise FileNotFoundError(f"{flood_events_path} not found.")

flood_events_df = pd.read_csv(flood_events_path, parse_dates=["Start", "End"])
allowed_sources = set(cfg.raw["data_sources"]["flood_events"]["sources_allowed"])
bad_rows = flood_events_df[~flood_events_df["Source"].isin(allowed_sources)]
if len(bad_rows) > 0:
    raise ValueError(f"Fix data/flood_events_basins.csv before proceeding.")
print(f"Loaded {len(flood_events_df)} verified flood events")

print("\n--- Step 4a: Building Initial Temporal & Rainfall Features ---")
df = rainfall_df.merge(points_df[["point_id", "lat", "lon", "basin_key"]], on=["point_id", "basin_key"])
df = df.sort_values(["point_id", "Date"]).reset_index(drop=True)
df = add_temporal_features(df)
df = add_rainfall_window_features(df, group_col="point_id")

# Compute rainfall climatology from TRAINING YEARS ONLY to prevent leakage
print("\n--- Step 4b: Computing Rainfall Climatology (train years only) ---")
train_years = cfg.raw["split"]["train_years"]
df_train_only = df[df["Date"].dt.year.isin(train_years)]
rainfall_climatology = compute_rainfall_climatology(df_train_only)
print(f"Climatology computed for {rainfall_climatology['basin_key'].nunique()} basins x "
      f"{rainfall_climatology['Day_of_Year'].nunique()} days-of-year")

# Add anomaly features to the FULL dataset (using train-derived climatology)
df = add_rainfall_anomaly_features(df, rainfall_climatology)

print(f"\nFeature matrix built: {df.shape}")
print(f"New anomaly columns: {[c for c in df.columns if 'Anomaly' in c or 'Wet_Flag' in c or 'Intensity' in c]}")

