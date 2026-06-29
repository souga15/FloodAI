"""
Spatial flood probability map for manuscript Figure.

Generates a publication-quality map showing:
  1. Mean predicted flood probability per grid point (test years 2023-2024)
  2. Actual flood event locations overlaid as red markers
  3. Basin bounding boxes as rectangles
  4. Color scale from low (blue) to high (red) probability

No shapefile or external basemap required — uses matplotlib only.
For higher-quality maps, optionally uses cartopy if available.

Output: /content/floodai_outputs/spatial_validation_map.png (300 DPI, TIFF for journals)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

logger = logging.getLogger("floodai.visualization.spatial_map")

# Publication-quality colormap: white -> yellow -> orange -> red
FLOOD_CMAP = LinearSegmentedColormap.from_list(
    "flood_risk",
    ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#f16913", "#cb181d"],
    N=256,
)


def generate_spatial_map(
    df: pd.DataFrame,
    test_proba: np.ndarray,
    points_df: pd.DataFrame,
    flood_events_df: pd.DataFrame,
    test_years: list[int],
    output_dir: str = "/content/floodai_outputs",
    feature_cols: list[str] | None = None,
) -> str:
    """
    Generate and save the spatial validation map.

    Args:
        df: full feature DataFrame with Date, point_id, basin_key, lat, lon
        test_proba: predicted probabilities for the test set rows (same order as df_test)
        points_df: point-level DataFrame with point_id, lat, lon, basin_key
        flood_events_df: flood events DataFrame with basin_key, Start, End columns
        test_years: list of test years (used to filter df for the map)
        output_dir: directory to save the figure
        feature_cols: feature columns used by model (for labelling)

    Returns:
        Path to saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build test-period predictions per point (mean probability)
    df_test = df[df["Date"].dt.year.isin(test_years)].copy()
    df_test = df_test.reset_index(drop=True)

    if len(df_test) != len(test_proba):
        raise ValueError(
            f"len(df_test)={len(df_test)} != len(test_proba)={len(test_proba)}. "
            "Ensure test_proba corresponds to df filtered by test_years."
        )

    df_test["flood_prob"] = test_proba

    # Mean predicted probability per point over test period
    point_mean = (
        df_test.groupby("point_id")["flood_prob"]
        .mean()
        .reset_index()
        .rename(columns={"flood_prob": "mean_flood_prob"})
    )

    # Join lat/lon
    point_map = point_mean.merge(
        points_df[["point_id", "lat", "lon", "basin_key"]], on="point_id", how="left"
    )

    # Flood event centroid locations for test years
    flood_test = flood_events_df[
        (pd.to_datetime(flood_events_df["Start"]).dt.year.isin(test_years)) |
        (pd.to_datetime(flood_events_df["End"]).dt.year.isin(test_years))
    ].copy()

    # Basin bounding boxes (from points)
    basin_boxes = (
        points_df.groupby("basin_key")
        .agg(lat_min=("lat", "min"), lat_max=("lat", "max"),
             lon_min=("lon", "min"), lon_max=("lon", "max"))
        .reset_index()
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    ax.set_facecolor("#e8f4f8")  # light blue background (ocean/land contrast)

    # Draw basin bounding boxes
    basin_colors = {
        "ganga_bihar":      "#2166ac",
        "brahmaputra_assam":"#4dac26",
        "mahanadi_odisha":  "#d01c8b",
        "sutlej_punjab":    "#f1a340",
    }
    basin_labels = {
        "ganga_bihar":      "Ganga (Bihar)",
        "brahmaputra_assam":"Brahmaputra (Assam)",
        "mahanadi_odisha":  "Mahanadi (Odisha)",
        "sutlej_punjab":    "Sutlej (Punjab)",
    }
    legend_patches = []
    for _, brow in basin_boxes.iterrows():
        key = brow["basin_key"]
        color = basin_colors.get(key, "#999999")
        rect = mpatches.FancyBboxPatch(
            (brow["lon_min"], brow["lat_min"]),
            brow["lon_max"] - brow["lon_min"],
            brow["lat_max"] - brow["lat_min"],
            boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor=color, facecolor="none", alpha=0.8,
            linestyle="--",
        )
        ax.add_patch(rect)
        # Label basin
        ax.text(
            (brow["lon_min"] + brow["lon_max"]) / 2,
            brow["lat_max"] + 0.15,
            basin_labels.get(key, key),
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
        )
        legend_patches.append(mpatches.Patch(edgecolor=color, facecolor="none",
                                              linestyle="--", linewidth=1.5,
                                              label=basin_labels.get(key, key)))

    # Scatter: grid points coloured by mean flood probability
    sc = ax.scatter(
        point_map["lon"], point_map["lat"],
        c=point_map["mean_flood_prob"],
        cmap=FLOOD_CMAP, vmin=0.0, vmax=1.0,
        s=30, alpha=0.85, linewidths=0.3, edgecolors="grey", zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.01, shrink=0.7)
    cbar.set_label("Mean Predicted Flood Probability\n(Test Period 2023–2024)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Overlay actual flood events as stars
    if len(flood_test) > 0 and "lat" in flood_test.columns and "lon" in flood_test.columns:
        ax.scatter(
            flood_test["lon"], flood_test["lat"],
            marker="*", s=200, color="#cc0000", edgecolors="white",
            linewidths=0.5, zorder=5, label="Verified Flood Event (CWC/DFO)",
        )
    else:
        # Approximate event location using basin centroid
        for _, ev in flood_test.iterrows():
            basin_row = basin_boxes[basin_boxes["basin_key"] == ev.get("Basin", ev.get("basin_key", ""))]
            if len(basin_row):
                clat = (float(basin_row["lat_min"]) + float(basin_row["lat_max"])) / 2
                clon = (float(basin_row["lon_min"]) + float(basin_row["lon_max"])) / 2
                ax.scatter(clon, clat, marker="*", s=180, color="#cc0000",
                           edgecolors="white", linewidths=0.5, zorder=5)

    legend_patches.append(mpatches.PathPatch(
        plt.scatter([], [], marker="*", s=100, color="#cc0000", label="Verified Flood Event (CWC/DFO)").get_paths()[0],
        color="#cc0000", label="Verified Flood Event (CWC/DFO)"
    ) if False else mpatches.Patch(facecolor="#cc0000", label="Verified Flood Event (CWC/DFO)"))

    # ── Axes / formatting ─────────────────────────────────────────────────────
    ax.set_xlim(72, 95)
    ax.set_ylim(18, 34)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title(
        "Physics-Informed XGBoost: Mean Predicted Flood Probability\n"
        "Four Indian River Basins — Validation Period 2023–2024",
        fontsize=13, fontweight="bold", pad=15,
    )

    # Grid lines
    ax.grid(True, linestyle=":", color="white", alpha=0.6, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)

    # Legend
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.85, edgecolor="grey")

    # Country outline: simple India bounding box annotation
    ax.annotate("India", xy=(81, 22), fontsize=14, color="#555555",
                alpha=0.3, fontweight="bold", ha="center")

    plt.tight_layout()
    out_png = output_dir / "spatial_validation_map.png"
    out_tif = output_dir / "spatial_validation_map_300dpi.tiff"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(out_tif, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("Spatial map saved: %s", out_png)
    return str(out_png)
