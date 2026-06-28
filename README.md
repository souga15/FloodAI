# FloodAI v0.1 — Basin-Scale Flood Occurrence Prediction

A modular, leakage-free, reproducible flood occurrence prediction framework
for three Indian river basins: Ganga (Bihar), Brahmaputra (Assam), and
Mahanadi (Odisha). Built as a deliberately narrower, more defensible
successor to an earlier 33-station single-notebook study (manuscript
WPT-D-26-00166, published in *Water Practice & Technology*).

## What changed from the prior project, and why

The prior notebook contained a cell (`Cell 23`) that bootstrap-resampled
metrics from the **full dataset including training rows**, producing
AUC=0.9815 / F1=0.7018 — numbers higher than the legitimate held-out test
result (AUC=0.9623 / F1=0.4878) precisely because the model had already seen
the resampled rows during training. The cell's own comments acknowledge this
("these numbers are HIGHER than test-set because training data is included")
while still naming the inflated figures as the manuscript's reported values.

This codebase makes that mistake structurally impossible to repeat:

- `floodai/evaluation/metrics.py` requires every `EvaluationResult` to carry
  a `DataProvenance` tag (`HELD_OUT`, `TRAINING_INCLUSIVE`, or
  `LOSO_HELD_OUT`).
- `report_headline()` — the only function permitted to label a number as a
  manuscript headline result — raises `HeadlineMetricError` if provenance
  is not `HELD_OUT`.
- `bootstrap_ci()` independently refuses to run if given
  `TRAINING_INCLUSIVE` provenance.
- `config/config.yaml`'s `reporting` section is validated at load time
  (`floodai/config.py`) to reject any configuration that would silently
  reintroduce a train-inclusive bootstrap as a headline source.
- `tests/test_metrics.py` and `tests/test_config.py` contain direct
  regression tests for this exact failure mode, named and documented as such.

This is not a stylistic preference. It is the central engineering
requirement of this project, and every other module is secondary to it.

## Second incident (first real pipeline run) — Year leakage + SMOTE ratio bug

A first real run of this pipeline (outside this development environment)
bypassed the `floodai` package's training modules and reimplemented the
training loop inline in a notebook cell. That reimplementation reproduced
two bugs the package was specifically designed to prevent, because it
didn't call into the modules that prevent them:

1. **`Year` leaked into the model.** `feature_cols` was built as
   `[c for c in df.columns if c not in exclude_cols]`, and `exclude_cols`
   did not include `Year`. With only ~10 verified flood events total, the
   model could use the literal year number as a near-perfect lookup key for
   "does this year contain a labeled flood" — a shortcut that has nothing
   to do with rainfall or terrain. This produced Leave-One-Basin-Out
   AUC=1.000 / Recall=1.000 *identically* across three basins with
   documented-different flood mechanisms, which is the signature of
   leakage, not generalization.
2. **SMOTE silently defaulted to 50/50 balancing.** The inline code called
   `SMOTE(random_state=cfg.random_seed)` with no `sampling_strategy`,
   instead of the ~10% specified in `config.yaml`. Confirmed by the printed
   output: `positives: 235280 / 470560` is exactly 50.0%.
3. **(Separately, a real data problem, not a code bug):** only 10 verified
   flood events existed across 3 basins x 8 years, and none fell in the
   2023-2024 test window, producing `ROC-AUC: nan` and `F1: 0.0000` on the
   held-out test set — mathematically correct given zero positive test
   samples, but a sign the flood-events CSV needs far more verified events
   before training is worth running at all.

**Fixes applied:**
- `floodai/features/governance.py` — `select_model_features()` is now the
  only sanctioned way to build a feature list; it allowlists known-safe
  feature groups (`Year` is explicitly never eligible) rather than
  excluding a manually maintained denylist. `assert_no_forbidden_columns()`
  is a second defensive check called inside `training/lobo.py` itself, so
  even a hand-rolled `feature_cols` list is caught before training.
- `floodai/training/imbalance.py` — `resample_training_only()` now verifies
  the post-SMOTE positive ratio matches the requested `sampling_strategy`
  within tolerance, and raises if it doesn't (catching exactly the silent
  50/50-default failure mode).
- `floodai/training/label_sufficiency.py` — `check_split_has_positives()`
  raises immediately after labelling if any split has too few positive
  samples, instead of letting 35 minutes of Optuna tuning run before
  discovering the test set was empty.
- `tests/test_governance.py`, `tests/test_imbalance.py`,
  `tests/test_label_sufficiency.py` — regression tests reproducing each
  failure mode directly.
- `notebooks/cell14_corrected.py` — a corrected version of the pipeline
  cell that calls into the package modules instead of reimplementing
  training inline. **Use this, not a hand-rolled version**, precisely
  because the hand-rolled version is what caused this incident.

## What is real vs. what requires your verification

**Built, tested, verified in this environment** (see `tests/`, all passing):
- Leakage-safe feature engineering (rolling windows, SCS-CN runoff)
- Provenance-tagged evaluation with the headline-metric guard
- SMOTE applied train-only by construction (function signature excludes
  val/test data)
- Optuna hyperparameter search driven entirely by config
- Leave-one-basin-out cross-validation
- Full pipeline wiring, proven via synthetic-data integration tests

**Written against documented APIs but NOT executed against a live external
service from this environment** (you must verify on first real run):
- `IMDGriddedRainfallProvider` (IMD gridded rainfall via `imdlib`) —
  `imdpune.gov.in` is outside this development sandbox's network access.
  Run `notebooks/validate_first_run.py` before trusting any downstream number.
- Administrative-boundary-based point generation
  (`gis.points.generate_admin_centroid_points`) requires a GADM/OSM India
  boundary file not bundled here; the driver notebook defaults to the
  deterministic grid fallback until you supply one.

**Explicitly not yet implemented (do not claim these in a manuscript until
built and validated):**
- Calibrated HAND-to-depth-in-meters conversion. `gis/terrain.py` computes
  an uncalibrated proxy and tags it `calibration_status="uncalibrated_
  literature_default"` specifically so it cannot be mistaken for a validated
  depth prediction. Calibration requires reference inundation data (e.g.
  Sentinel-1 SAR flood extents) not yet acquired.
- Multi-horizon (1/3/5-day-ahead) forecasting.
- Deep learning baselines (LSTM/GRU/TFT/TabNet).

## Data requirements you must supply

`data/flood_events_basins.csv` does not exist yet. It must contain verified
flood events with `Source` restricted to `CWC`, `DFO`, or `EM-DAT` —
"News"-sourced events from the prior project's dataset are deliberately
excluded pending independent verification (see `config.yaml` comments).

## Second real-run result and the open Mahanadi question

A corrected run (using `cell14_corrected.py`, with 20 verified events instead
of 10) produced:

- **Held-out test (2023-2024)**: ROC-AUC=0.8949, F1=0.6204, MCC=0.5627,
  PR-AUC=0.5545 — approved via `report_headline()`, provenance=held_out.
- **LOBO**: AUC≈0.91 across all three basins (0.912 Brahmaputra, 0.915
  Ganga, 0.912 Mahanadi) with genuine F1 variation (0.49–0.53) — this is the
  believable pattern the Year-leakage fix predicted, replacing the previous
  spurious AUC=1.000-everywhere result.

This run used **rainfall and calendar features only** — terrain/CN/TWI joins
were not yet wired in (see `gis/terrain.py` status). It is a checkpoint, not
the final model.

**Open question before reporting the Mahanadi LOBO number**: Mahanadi is
documented (in the prior WPT-D-26-00166 study and in `config.yaml`'s
`tier_prior`) as a reservoir/cyclone-driven basin expected to be poorly
explained by local rainfall. Scoring nearly identically to two genuinely
rainfall-driven basins is surprising enough to need explicit investigation
before it goes in a manuscript. Two explanations are both consistent with
the data so far:

  (a) The model legitimately predicts a rainfall-correlated subset of
      Mahanadi floods (e.g. cyclone landfall events that also bring heavy
      local rain even though the *primary* flood mechanism is surge or
      reservoir release).
  (b) The curated event list for Mahanadi is itself biased toward
      well-documented, rainfall-coincident events, because those are easier
      to find independent CWC/DFO/EM-DAT citations for — a labeling
      artifact, not a model capability.

`floodai/evaluation/label_diagnostics.py` (`compute_event_rainfall_context`,
`compare_basins_rainfall_coincidence`) computes the event-by-event evidence
needed to tell these apart — for each verified flood event, where does its
rainfall fall relative to the basin's own non-flood baseline? Run
`notebooks/diagnose_mahanadi_labels.py` against your real `df` and
`flood_events_df` and read its printed guidance before writing up the
Mahanadi result either way.



```
config/config.yaml          All experiment parameters — edit here, not in src/
src/floodai/
  config.py                 Validated config loading
  logging_utils.py          Run manifests + logging
  data/                     Rainfall/DEM provider interfaces + IMD/NASA-POWER impls
  gis/                      Basin point generation, terrain (slope/TWI/HAND)
  features/                 Leakage-safe feature engineering
  models/                   XGBoost construction (correct early-stopping API)
  training/                 SMOTE (train-only), Optuna tuning, threshold, LOBO
  evaluation/                Provenance-tagged metrics + headline guard
notebooks/
  01_run_pipeline.ipynb      Thin orchestration notebook — no logic lives here
  validate_first_run.py     MANDATORY first-run data sanity check
tests/                       23 tests, all passing as of this build
```

## Running

```bash
pip install -r requirements.txt
pip install -e .
pytest tests/ -v                          # should show 23 passed
python notebooks/validate_first_run.py    # MANDATORY before real data use
```

Then open `notebooks/01_run_pipeline.ipynb` in Colab.
