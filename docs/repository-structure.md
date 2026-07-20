# Repository Structure

This repo is intentionally a script workspace. The goal is to make each file's
purpose obvious at a glance without forcing a package layout before the scripts
have stabilized.

## Standard Directories

| Path | Purpose |
| --- | --- |
| `scripts/earthengine/` | Python entry points that authenticate with Earth Engine, submit exports, sample imagery, or train models from Earth Engine-derived data. |
| `scripts/cli/` | Local command-line utilities for CSV cleaning, plotting, clipping, or data inspection that do not need Earth Engine. |
| `earthengine/code-editor/` | JavaScript files intended for the Google Earth Engine Code Editor UI. |
| `data/raw/` | Small source inputs that can reasonably be versioned. |
| `data/processed/` | Derived tables, sampled feature caches, and other reproducible intermediate data. |
| `data/external/` | Downloaded third-party reference layers used locally. |
| `outputs/figures/` | Generated figures and exploratory plots. |
| `outputs/models/` | Trained model artifacts, metrics, and model diagnostics. |
| `secrets/` | Local service account keys and credentials. |
| `docs/` | Project notes, history, and repo maintenance documentation. |

## Script Inventory

| Script | Role |
| --- | --- |
| `scripts/earthengine/export_layers.py` | Large Earth Engine data-pull/export runner. Future committed export plugins should land here or in a sibling module under `scripts/earthengine/`. |
| `scripts/earthengine/sample_embeddings_and_train_models.py` | Samples annual Google Satellite Embedding bands for point observations and trains target models. |
| `scripts/earthengine/sample_multisensor_bands_and_train_models.py` | Samples Sentinel-2, Sentinel-1, and Landsat bands around point-observation dates and trains target models. |
| `scripts/cli/clip_aim_table_to_bbox.py` | Filters an AIM-style CSV to a bounding box around its centroid. |
| `scripts/cli/plot_point_distribution.py` | Plots AIM point locations over a country boundary layer. |
| `scripts/cli/plot_example_area.py` | Plots a supplied WKT polygon as a quick area sanity check. |
| `earthengine/code-editor/conceptual-framework-explorer.js` | Interactive split-map explorer for conceptual framework predictor layers. |
| `earthengine/code-editor/pca-analysis.js` | Interactive drawn-region PCA analysis over predictor layers. |
| `earthengine/code-editor/grasslands-integrity-reference-site.js` | Reference-site mask construction script. |
| `earthengine/code-editor/grasslands-integrity-thresholds-ui.js` | Threshold-tuning UI for grassland probability, HII, and HMI masks. |

## Future Additions

- Put the planned larger Earth Engine data-pull script in
  `scripts/earthengine/`. If it grows past one file, split reusable layer
  definitions into `scripts/earthengine/layers/` while keeping a clear runnable
  entry point.
- Put one-off local utilities in `scripts/cli/` with imperative names such as
  `clip_table.py`, `plot_point_distribution.py`, or `summarize_outputs.py`.
- Keep Code Editor experiments in `earthengine/code-editor/`; use descriptive
  filenames that name the workflow rather than the date of the experiment.
- Treat `data/processed/`, `data/external/`, and `outputs/` as reproducible
  local working areas unless a file is small, stable, and intentionally useful
  as an example.
