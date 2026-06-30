import sys
from pathlib import Path
sys.path.insert(0, "src")

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from floodai.config import load_config
from floodai.gis.points import generate_grid_fallback_points, basin_points_to_dataframe
from floodai.data.rainfall_providers import get_rainfall_provider
from floodai.features.pipeline import add_temporal_features, add_rainfall_window_features, add_scs_cn_runoff, add_interaction_features
from floodai.features.governance import select_model_features, assert_no_forbidden_columns
from floodai.evaluation.metrics import DataProvenance, evaluate, report_headline
from floodai.models.xgb_model import build_xgb_classifier, fit_with_validation
from floodai.training.imbalance import resample_training_only
from floodai.training.label_sufficiency import check_basin_has_positives, check_split_has_positives
from floodai.training.lobo import run_lobo_cv
from floodai.training.threshold import select_f1_optimal_threshold
from floodai.training.tuning import run_optuna_search
from floodai.gis.terrain_real import add_real_terrain_features

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Load config
cfg = load_config(Path("config/config.yaml"))
output_dir = Path(cfg.raw["experiment"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Point generation
print("--- Step 1: Generating points ---")
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

# 2. Rainfall Ingestion
print("\n--- Step 2: Fetching/Loading Rainfall Data ---")
provider = get_rainfall_provider(
    cfg.raw["data_sources"]["rainfall"]["provider"],
)
start_year = cfg.raw["data_sources"]["rainfall"]["start_year"]
end_year = cfg.raw["data_sources"]["rainfall"]["end_year"]

# Check if raw rainfall is cached
rainfall_cache_path = output_dir / "rainfall_raw.parquet"
if rainfall_cache_path.exists():
    print(f"[CACHE HIT] Loading rainfall from {rainfall_cache_path}")
    rainfall_df = pd.read_parquet(rainfall_cache_path)
else:
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
            pass # Gracefully skip grid points outside land boundaries
        if (i + 1) % 20 == 0 or (i + 1) == len(points_df):
            print(f"  {i+1}/{len(points_df)} points fetched...")
    rainfall_df = pd.concat(all_series, ignore_index=True)
    rainfall_df.to_parquet(rainfall_cache_path, index=False)
    print(f"Saved rainfall raw cache to {rainfall_cache_path}")

# 3. Base Features
print("\n--- Step 3: Building temporal & rainfall features ---")
df = rainfall_df.merge(points_df[["point_id", "lat", "lon", "basin_key"]], on=["point_id", "basin_key"])
df = df.sort_values(["point_id", "Date"]).reset_index(drop=True)
df = add_temporal_features(df)
df = add_rainfall_window_features(df, group_col="point_id")

# 4. Terrain Join
print("\n--- Step 4: Joining real terrain features (Elevation, CN, TWI) ---")
terrain_cache_path = output_dir / "terrain_cache.parquet"
if terrain_cache_path.exists():
    print(f"[CACHE HIT] Loading terrain from {terrain_cache_path} — skipping API calls.")
    terrain_cols = ["point_id", "Elevation_m", "Curve_Number", "TWI"]
    terrain_cache_df = pd.read_parquet(terrain_cache_path)
    df = df.drop(columns=[c for c in terrain_cols[1:] if c in df.columns], errors="ignore")
    df = df.merge(terrain_cache_df[terrain_cols], on="point_id", how="left")
    df = add_scs_cn_runoff(df)
    df = add_interaction_features(df)
    print("[OK] Terrain restored from cache.")
else:
    print("Computing real terrain features (ISRIC SoilGrids + Open-Elevation)...")
    # This calls add_real_terrain_features which fetches SoilGrids CN and computes pysheds TWI
    df = add_real_terrain_features(df, points_df, dem_cache_dir=str(output_dir / "dem_cache"))
    terrain_to_cache = df[["point_id", "Elevation_m", "Curve_Number", "TWI"]].drop_duplicates("point_id")
    terrain_to_cache.to_parquet(terrain_cache_path, index=False)
    print(f"[OK] Terrain cached to {terrain_cache_path} for future sessions.")

# 5. Label Floods
print("\n--- Step 5: Loading and applying flood event labels ---")
flood_events_path = Path(cfg.raw["data_sources"]["flood_events"]["path"])
flood_events_df = pd.read_csv(flood_events_path, parse_dates=["Start", "End"])
flood_events_df = flood_events_df.rename(columns={"Basin": "basin_key"})

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
vc = df['Flood_Occurred'].value_counts()
print(f"Flood label distribution:\n{vc}")
print(f"Positive rate: {vc.get(1,0)/len(df)*100:.2f}%")

# 6. Checks
check_split_has_positives(
    df, date_col='Date', label_col='Flood_Occurred',
    train_years=cfg.raw['split']['train_years'],
    val_years=cfg.raw['split']['val_years'],
    test_years=cfg.raw['split']['test_years'],
    min_positives_per_split=5,
)
basin_counts = check_basin_has_positives(df, basin_col='basin_key', label_col='Flood_Occurred')
print(f"Per-basin positive counts: {basin_counts}")

# 7. Split
train_years = cfg.raw['split']['train_years']
val_years   = cfg.raw['split']['val_years']
test_years  = cfg.raw['split']['test_years']

df_train = df[df['Date'].dt.year.isin(train_years)].copy()
df_val   = df[df['Date'].dt.year.isin(val_years)].copy()
df_test  = df[df['Date'].dt.year.isin(test_years)].copy()

# Governed features - year and other forbidden features are excluded by construction
feature_groups = ["rainfall_current", "rainfall_windows", "rainfall_anomaly", "terrain_physics", "interaction"]
feature_cols = select_model_features(df, groups=feature_groups)
assert_no_forbidden_columns(feature_cols)
print(f"\nSelected {len(feature_cols)} governed features:")
print(feature_cols)

X_train, y_train = df_train[feature_cols].values, df_train['Flood_Occurred'].values
X_val,   y_val   = df_val[feature_cols].values,   df_val['Flood_Occurred'].values
X_test,  y_test  = df_test[feature_cols].values,  df_test['Flood_Occurred'].values

# 8. Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# 9. Resample (SMOTE)
imb_cfg = cfg.raw['imbalance']
X_train_res, y_train_res = resample_training_only(
    X_train_scaled, y_train,
    sampling_strategy=imb_cfg['sampling_strategy'],
    k_neighbors_max=imb_cfg['k_neighbors_max'],
    seed=cfg.random_seed,
)

# 10. Tuning (Optuna)
search_space = cfg.raw['model']['optuna']['search_space']
n_trials = cfg.raw['model']['optuna']['n_trials']
early_stopping_rounds = cfg.raw['model']['early_stopping_rounds']

best_params = run_optuna_search(
    X_train_res, y_train_res, X_val_scaled, y_val,
    search_space=search_space, n_trials=n_trials,
    early_stopping_rounds=early_stopping_rounds, seed=cfg.random_seed,
)
print(f"\nBest params: {best_params}")

best_model = build_xgb_classifier(best_params, early_stopping_rounds, cfg.random_seed)
best_model = fit_with_validation(best_model, X_train_res, y_train_res, X_val_scaled, y_val)

# 11. Threshold selection
val_proba = best_model.predict_proba(X_val_scaled)[:, 1]
tau_star = select_f1_optimal_threshold(y_val, val_proba)

# 12. Evaluate
test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
result = evaluate(
    y_test, test_proba, threshold=tau_star,
    set_name=f"test_{test_years[0]}_{test_years[-1]}",
    provenance=DataProvenance.HELD_OUT,
)
headlined = report_headline(result)

print("\n" + "="*55)
print("  HELD-OUT TEST SET RESULTS (headline-approved)")
print("="*55)
print(f"  ROC-AUC   : {headlined.roc_auc:.4f}")
print(f"  PR-AUC    : {headlined.pr_auc:.4f}")
print(f"  F1 Score  : {headlined.f1:.4f}")
print(f"  MCC       : {headlined.mcc:.4f}")
print(f"  Bal. Acc  : {headlined.balanced_accuracy:.4f}")
print(f"  FAR       : {headlined.far:.4f}")
print(f"  CSI       : {headlined.csi:.4f}")
print(f"  Threshold : {headlined.threshold:.4f}")
tn, fp, fn, tp = headlined.confusion
print(f"  TN={tn:>6}  FP={fp:>6}  FN={fn:>6}  TP={tp:>6}")
print("="*55)

# 13. LOBO
print("\nRunning LOBO cross-validation (governed)...")
lobo_results = run_lobo_cv(
    df, feature_columns=feature_cols, target_column='Flood_Occurred',
    basin_column='basin_key', best_params=best_params,
    early_stopping_rounds=early_stopping_rounds,
    smote_sampling_strategy=imb_cfg['sampling_strategy'],
    smote_k_neighbors_max=imb_cfg['k_neighbors_max'],
    seed=cfg.random_seed,
)

lobo_df = pd.DataFrame([{
    'held_out_basin': r.set_name.replace('LOBO_held_out_', ''),
    'ROC_AUC': r.roc_auc, 'PR_AUC': r.pr_auc, 'F1': r.f1, 'MCC': r.mcc,
    'n_test': r.n_samples, 'n_positive': r.n_positive,
} for r in lobo_results])

lobo_df.to_csv(output_dir / "lobo_results.csv", index=False)

print("\n" + "="*65)
print("  LOBO CV RESULTS")
print("="*65)
print(lobo_df.to_string(index=False))
if len(lobo_df) > 0:
    print("-"*65)
    print(f"  Mean ROC-AUC : {lobo_df['ROC_AUC'].mean():.4f} ± {lobo_df['ROC_AUC'].std():.4f}")
    print(f"  Mean F1      : {lobo_df['F1'].mean():.4f} ± {lobo_df['F1'].std():.4f}")
    print(f"  Mean MCC     : {lobo_df['MCC'].mean():.4f} ± {lobo_df['MCC'].std():.4f}")
    print("="*65)
    print(f"Saved: {output_dir / 'lobo_results.csv'}")

# 14. Mahanadi Diagnostic
print("\n--- Running Mahanadi Diagnostic ---")
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
