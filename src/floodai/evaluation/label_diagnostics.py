"""
Label-rainfall coincidence diagnostics.

Why this exists: a LOBO run on Mahanadi (a basin documented and expected to
be reservoir/cyclone-driven, i.e. NOT well-explained by local rainfall
alone) scored AUC=0.912 -- nearly identical to Ganga (0.915) and
Brahmaputra (0.912), which are genuinely rainfall-driven. That similarity
is surprising enough to investigate before reporting it as a finding. Two
explanations are possible, and they lead to opposite conclusions in a
manuscript:

  (a) The model is correctly catching a rainfall-correlated SUBSET of
      Mahanadi floods (e.g. cyclone landfall events that also dump heavy
      local rain even though the flood mechanism is technically
      surge/reservoir-driven). This is a genuine, reportable finding.
  (b) The curated event list for Mahanadi is itself biased toward
      well-documented, rainfall-coincident events (because those are
      easier to find independent CWC/DFO/EM-DAT citations for), while
      "quiet-rainfall" reservoir-release floods were under-sampled into
      the label set. This is a labeling artifact, not a model capability,
      and must be disclosed as a limitation rather than presented as
      evidence the model generalizes to reservoir-driven mechanisms.

This module does not decide between (a) and (b) -- it produces the
diagnostic evidence (event-by-event rainfall context) that a human needs to
make that call, since it requires judgment about each specific historical
event that no purely statistical threshold can substitute for.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("floodai.evaluation.label_diagnostics")


def compute_event_rainfall_context(
    df: pd.DataFrame,
    flood_events_df: pd.DataFrame,
    basin_key: str,
    rainfall_col: str = "Rainfall_7Day_mm",
    date_col: str = "Date",
    basin_col: str = "basin_key",
) -> pd.DataFrame:
    """
    For every verified flood event in `basin_key`, compute the distribution
    of 7-day cumulative rainfall (or another rainfall feature) across all
    points/days that fall inside that event's window, and compare it to the
    basin's non-flood baseline. This is the event-by-event evidence needed
    to judge whether labeled floods are rainfall-coincident.

    Returns one row per event with:
      - event window dates, severity, source (for traceability back to the CSV)
      - mean/median/max rainfall_col during the event window
      - the basin's non-flood baseline median, for direct comparison
      - a `rainfall_percentile_vs_baseline` score: where the event's median
        rainfall falls within the basin's overall (non-flood) rainfall
        distribution. A LOW percentile here is the concrete evidence for
        explanation (b) above -- a flood event with unremarkable rainfall.
    """
    basin_df = df[df[basin_col] == basin_key].copy()
    basin_events = flood_events_df[flood_events_df[basin_col] == basin_key].copy() if basin_col in flood_events_df.columns else flood_events_df.copy()

    if len(basin_events) == 0:
        raise ValueError(f"No flood events found for basin '{basin_key}' in flood_events_df.")

    non_flood_mask = basin_df["Flood_Occurred"] == 0 if "Flood_Occurred" in basin_df.columns else pd.Series(True, index=basin_df.index)
    baseline_rainfall = basin_df.loc[non_flood_mask, rainfall_col]
    baseline_median = baseline_rainfall.median()

    rows = []
    for _, ev in basin_events.iterrows():
        window_mask = (basin_df[date_col] >= ev["Start"]) & (basin_df[date_col] <= ev["End"])
        window_rain = basin_df.loc[window_mask, rainfall_col]

        if len(window_rain) == 0:
            logger.warning(
                "Event %s (%s to %s) has zero matching rows in basin '%s' -- "
                "check date alignment / point coverage for this window.",
                ev.get("Region_Name", "?"), ev["Start"].date(), ev["End"].date(), basin_key,
            )
            rows.append({
                "basin": basin_key, "start": ev["Start"], "end": ev["End"],
                "severity": ev.get("Severity"), "source": ev.get("Source"),
                "n_rows": 0, "mean_rain": np.nan, "median_rain": np.nan, "max_rain": np.nan,
                "baseline_median": baseline_median, "rainfall_percentile_vs_baseline": np.nan,
            })
            continue

        percentile = float((baseline_rainfall < window_rain.median()).mean() * 100)
        rows.append({
            "basin": basin_key, "start": ev["Start"], "end": ev["End"],
            "severity": ev.get("Severity"), "source": ev.get("Source"),
            "n_rows": len(window_rain),
            "mean_rain": float(window_rain.mean()),
            "median_rain": float(window_rain.median()),
            "max_rain": float(window_rain.max()),
            "baseline_median": float(baseline_median),
            "rainfall_percentile_vs_baseline": percentile,
        })

    result = pd.DataFrame(rows).sort_values("rainfall_percentile_vs_baseline")
    n_low = (result["rainfall_percentile_vs_baseline"] < 50).sum()
    logger.info(
        "Basin '%s': %d/%d events have BELOW-MEDIAN rainfall relative to "
        "the basin's own non-flood baseline. A high count here means many "
        "labeled events are NOT rainfall-coincident, which would argue "
        "against explanation (b) (label bias toward rainfall-heavy events) "
        "and is itself worth reporting as a model-capability question -- "
        "if the model still scores well on these, that's a different, "
        "more surprising finding than 'it found the rainfall-correlated subset'.",
        basin_key, n_low, len(result),
    )
    return result


def compare_basins_rainfall_coincidence(
    df: pd.DataFrame,
    flood_events_df: pd.DataFrame,
    basin_keys: list[str],
    rainfall_col: str = "Rainfall_7Day_mm",
) -> pd.DataFrame:
    """
    Side-by-side summary across basins: for each basin, what fraction of its
    labeled flood events are "rainfall-coincident" (event-window median
    rainfall at or above the basin's own non-flood median)?

    If Mahanadi's fraction is similar to Ganga/Brahmaputra's, that is
    concrete, basin-specific evidence supporting explanation (a) or (b)
    above (the diagnostic doesn't distinguish them on its own -- a human
    needs to look at the per-event table from compute_event_rainfall_context
    to tell which). If Mahanadi's fraction is much LOWER but LOBO AUC was
    still high, that's the more interesting (and more surprising) result,
    worth a deeper look at which features are actually driving Mahanadi
    predictions (e.g. via SHAP) before trusting the LOBO score there.
    """
    summary_rows = []
    for basin_key in basin_keys:
        try:
            event_table = compute_event_rainfall_context(df, flood_events_df, basin_key, rainfall_col)
        except ValueError as e:
            logger.warning("Skipping basin '%s': %s", basin_key, e)
            continue
        valid = event_table.dropna(subset=["rainfall_percentile_vs_baseline"])
        frac_coincident = float((valid["rainfall_percentile_vs_baseline"] >= 50).mean()) if len(valid) else np.nan
        summary_rows.append({
            "basin": basin_key,
            "n_events": len(event_table),
            "n_events_with_data": len(valid),
            "fraction_rainfall_coincident": frac_coincident,
            "median_event_percentile": float(valid["rainfall_percentile_vs_baseline"].median()) if len(valid) else np.nan,
        })

    summary = pd.DataFrame(summary_rows)
    logger.info("Cross-basin rainfall-coincidence summary:\n%s", summary.to_string(index=False))
    return summary
