# NHI Grasslands

Scripts and Google Earth Engine experiments for building and exploring grassland
reference-site and predictor layers.

The repository is organized around a small number of runnable scripts rather
than a Python package. Run commands from the repository root so the relative
`data/`, `outputs/`, and `secrets/` paths resolve consistently.

## Layout

```text
earthengine/code-editor/    Earth Engine JavaScript scripts for the Code Editor.
scripts/earthengine/        Python scripts that use the Earth Engine API.
scripts/cli/                Local command-line utilities that do not need Earth Engine.
data/raw/                   Small source inputs that are okay to version.
data/processed/             Local derived tables and caches, ignored by git.
data/external/              Local downloaded reference data, ignored by git.
outputs/                    Generated figures, metrics, and model artifacts.
secrets/                    Local credentials, ignored by git.
docs/                       Notes, history, and repo-layout documentation.
```

See [docs/repository-structure.md](docs/repository-structure.md) for the script
inventory and the intended home for future data-pull work.

## Current Scripts

- [scripts/earthengine/export_layers.py](scripts/earthengine/export_layers.py)
  starts Earth Engine exports to Cloud Storage and skips tasks that already
  exist.
- [scripts/earthengine/sample_embeddings_and_train_models.py](scripts/earthengine/sample_embeddings_and_train_models.py)
  samples annual Satellite Embedding bands at AIM points and trains target
  models.
- [scripts/earthengine/sample_multisensor_bands_and_train_models.py](scripts/earthengine/sample_multisensor_bands_and_train_models.py)
  samples Sentinel-2, Sentinel-1, and Landsat bands near observation dates and
  trains target models.
- [scripts/cli/clip_aim_table_to_bbox.py](scripts/cli/clip_aim_table_to_bbox.py)
  filters an AIM-style CSV to a centroided bounding box.
- [scripts/cli/plot_point_distribution.py](scripts/cli/plot_point_distribution.py)
  writes a point-distribution map from AIM coordinates.
- [scripts/cli/plot_example_area.py](scripts/cli/plot_example_area.py)
  writes a quick EPSG:4326 polygon plot from WKT.
- [earthengine/code-editor/conceptual-framework-explorer.js](earthengine/code-editor/conceptual-framework-explorer.js)
  provides the split-map predictor explorer.
- [earthengine/code-editor/pca-analysis.js](earthengine/code-editor/pca-analysis.js)
  runs a drawn-region PCA over predictor layers.
- [earthengine/code-editor/grasslands-integrity-reference-site.js](earthengine/code-editor/grasslands-integrity-reference-site.js)
  builds the reference-site integrity masks.
- [earthengine/code-editor/grasslands-integrity-thresholds-ui.js](earthengine/code-editor/grasslands-integrity-thresholds-ui.js)
  exposes threshold controls for grassland, HII, and HMI layers.

## Earth Engine Shell

Build the Docker image and open a shell with the repository mounted:

```bat
docker build -t gee_env .
docker run --rm -it -v %CD%:/workdir gee_env
```

The Python Earth Engine scripts expect a local service account key at
`secrets/service-account-key.json`. Do not commit credentials.

## Data And Outputs

Keep small, stable source inputs in `data/raw/`. Put generated CSVs, Parquet
caches, GeoPackages, trained models, metrics, and plots in `data/processed/`,
`data/external/`, or `outputs/`; those locations are ignored by default.
