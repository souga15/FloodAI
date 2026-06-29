"""
Step 9 — Spatial Validation Map
Run AFTER step8b (baselines + SHAP). Requires test_proba and points_df in memory.
Saves publication-quality spatial map to /content/floodai_outputs/
"""
import sys, importlib
sys.path.insert(0, "/content/FloodAI/src")

import floodai.visualization.spatial_map as _sm
importlib.reload(_sm)
from floodai.visualization.spatial_map import generate_spatial_map

print("Generating spatial validation map...")
print("(No shapefiles required — pure matplotlib)")

try:
    out_path = generate_spatial_map(
        df=df,
        test_proba=test_proba,
        points_df=points_df,
        flood_events_df=flood_events_df,
        test_years=cfg.raw['split']['test_years'],
        output_dir="/content/floodai_outputs",
    )
    print(f"\n[OK] Map saved: {out_path}")
    print(f"[OK] 300 DPI TIFF (for journal submission): {out_path.replace('.png', '_300dpi.tiff')}")

    # Display inline in Colab
    from IPython.display import Image, display
    display(Image(filename=out_path))

except NameError as e:
    print(f"Missing variable: {e}")
    print("Make sure test_proba, df, points_df, flood_events_df are in memory from Step 7.")
except Exception as e:
    print(f"Map generation failed: {e}")
    import traceback; traceback.print_exc()
