# Flood Events Schema — `data/flood_events_basins.csv`

This file must be created before running the pipeline past Step 3. It is not
auto-generated, because flood event verification requires human judgment
about source reliability — that is precisely why `config.yaml` restricts
`sources_allowed` to `CWC`, `DFO`, `EM-DAT` rather than accepting anything.

## Required columns

| Column | Type | Description |
|---|---|---|
| `Basin` | str | One of: `ganga_bihar`, `brahmaputra_assam`, `mahanadi_odisha` |
| `Region_Name` | str | District/sub-district name affected (free text, for human readability) |
| `Start` | date (YYYY-MM-DD) | First day of the flood event window |
| `End` | date (YYYY-MM-DD) | Last day of the flood event window |
| `Severity` | str | One of: `Low`, `Medium`, `High`, `Very High`, `Extreme` |
| `Source` | str | One of: `CWC`, `DFO`, `EM-DAT` (enforced at load time — see `config.yaml`) |
| `Source_Reference` | str | Specific citation/URL/report ID for traceability — required for CWC entries (cite the specific annual report and page/table), required for DFO (cite the DFO event ID), required for EM-DAT (cite the EM-DAT disaster number) |

## Why "News" sources are excluded here

The prior project's dataset (WPT-D-26-00166) used `News` as an allowed
source for roughly a third of its flood events. A single newspaper report is
not independently re-verifiable the way a CWC gauge reading, a DFO satellite
detection, or an EM-DAT disaster-database entry is. For a paper targeting the
*International Journal of Disaster Risk Reduction*, every event in the
ground-truth label set should be traceable to a primary source a reviewer
could independently check. If you have specific newspaper-sourced events you
believe are well-corroborated, the recommended path is to find the
corresponding CWC/DFO/EM-DAT record (most major events have one) rather than
relaxing this constraint.

## Minimum viable event count

Each basin needs enough verified events across the 2017-2024 study period to
support the temporal train (2017-2020) / val (2021-2022) / test (2023-2024)
split with a non-zero positive class in every split. If a basin has too few
verified CWC/DFO/EM-DAT events to populate all three splits, that is itself
a finding worth reporting (e.g. for Mahanadi/Odisha, where you've already
flagged F1=0.000 as an expected hard case) — do not pad the event list with
weaker sources just to avoid an empty split.
