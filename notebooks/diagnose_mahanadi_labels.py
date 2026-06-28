# Diagnostic: is the Mahanadi LOBO result (AUC=0.912, nearly identical to
# Ganga/Brahmaputra) real generalization, or rainfall-coincident label bias?
#
# Run this AFTER Step 5 (label_floods) and Step 6 (feature engineering)
# from the corrected pipeline cell -- it needs `df` (the full feature
# dataframe with Flood_Occurred already labeled) and `flood_events_df`
# (loaded in Step 3) to both be in scope.
#
# What to look for in the output:
#   - `event_table_mahanadi`: one row per verified Mahanadi flood event,
#     sorted by rainfall_percentile_vs_baseline (ascending). Events near
#     the TOP of the sorted table (high percentile) were genuinely
#     rainfall-heavy. Events near the BOTTOM (low percentile, e.g. <30)
#     occurred during unremarkable or below-average rainfall for that
#     basin -- exactly the "quiet reservoir release" pattern the original
#     WPT paper's Tier-3 framing predicts.
#   - `coincidence_summary`: the headline comparison number. If Mahanadi's
#     fraction_rainfall_coincident is close to Ganga/Brahmaputra's, that is
#     evidence (not proof) that the model's Mahanadi performance comes from
#     a rainfall-correlated SUBSET of events, which is a fine and reportable
#     finding -- but the manuscript text should say exactly that ("the model
#     predicts the rainfall-coincident subset of Mahanadi floods") rather
#     than implying it generalizes to reservoir-release mechanisms broadly.
#   - If fraction_rainfall_coincident is LOW for Mahanadi (most events are
#     rainfall-quiet) but LOBO AUC was still ~0.91, that is the more
#     surprising result and needs a SHAP check on the Mahanadi fold before
#     trusting it -- it would mean the model found some other signal, and
#     you need to know what that signal is before reporting it.

from floodai.evaluation.label_diagnostics import (
    compare_basins_rainfall_coincidence,
    compute_event_rainfall_context,
)

basin_keys = list(cfg.basins.keys())  # ['ganga_bihar', 'brahmaputra_assam', 'mahanadi_odisha']

print("="*70)
print("  MAHANADI EVENT-BY-EVENT RAINFALL CONTEXT")
print("="*70)
event_table_mahanadi = compute_event_rainfall_context(
    df, flood_events_df, basin_key="mahanadi_odisha", rainfall_col="Rainfall_7Day_mm",
)
print(event_table_mahanadi.to_string(index=False))

print("\n" + "="*70)
print("  CROSS-BASIN RAINFALL-COINCIDENCE COMPARISON")
print("="*70)
coincidence_summary = compare_basins_rainfall_coincidence(
    df, flood_events_df, basin_keys=basin_keys, rainfall_col="Rainfall_7Day_mm",
)
print(coincidence_summary.to_string(index=False))

print("\n" + "="*70)
mahanadi_frac = coincidence_summary.loc[
    coincidence_summary["basin"] == "mahanadi_odisha", "fraction_rainfall_coincident"
].iloc[0]
other_fracs = coincidence_summary.loc[
    coincidence_summary["basin"] != "mahanadi_odisha", "fraction_rainfall_coincident"
]
print(f"Mahanadi rainfall-coincident fraction : {mahanadi_frac:.2%}")
print(f"Other basins (mean)                   : {other_fracs.mean():.2%}")
if abs(mahanadi_frac - other_fracs.mean()) < 0.15:
    print("\n-> Mahanadi's events are about as rainfall-coincident as the other")
    print("   two basins. This SUPPORTS explanation (a): the model is likely")
    print("   predicting a genuinely rainfall-correlated subset of Mahanadi")
    print("   floods. Report the LOBO result, but state explicitly in the")
    print("   manuscript that it reflects the rainfall-coincident subset of")
    print("   events, not the full space of reservoir-release flood triggers.")
else:
    print("\n-> Mahanadi's events are noticeably LESS rainfall-coincident than")
    print("   the other basins, yet LOBO AUC was still ~0.91. This is the more")
    print("   surprising result. Before reporting it: run SHAP on the Mahanadi")
    print("   LOBO fold specifically and check which features actually drove")
    print("   its predictions -- do not assume it's rainfall-driven generalization")
    print("   just because the AUC number looks good.")
