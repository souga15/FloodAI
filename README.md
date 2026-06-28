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

## Repository layout

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
