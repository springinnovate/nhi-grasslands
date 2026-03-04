import logging
import os
import time

import ee
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
)
log = logging.getLogger(__name__)


def write_actual_vs_predicted_scatterplots(
    embeddings_table: pd.DataFrame,
    embedding_band_names: list[str],
    target_column_names: list[str],
    models_output_directory: str,
    plots_output_directory: str,
    train_test_seed: int,
) -> None:
    os.makedirs(plots_output_directory, exist_ok=True)

    feature_matrix = embeddings_table[embedding_band_names].to_numpy(
        dtype=float
    )
    sample_count = len(embeddings_table)
    all_row_indices = np.arange(sample_count)

    train_row_indices, test_row_indices = train_test_split(
        all_row_indices,
        test_size=0.2,
        random_state=train_test_seed,
    )

    for target_name in tqdm(target_column_names, desc="Plots", unit="plot"):
        model_path = os.path.join(
            models_output_directory, f"ridge_{target_name}.joblib"
        )
        model_pipeline = joblib.load(model_path)

        actual_values = embeddings_table[target_name].to_numpy(dtype=float)
        predicted_values = model_pipeline.predict(feature_matrix)

        actual_train_values = actual_values[train_row_indices]
        predicted_train_values = predicted_values[train_row_indices]

        actual_test_values = actual_values[test_row_indices]
        predicted_test_values = predicted_values[test_row_indices]

        train_r2 = r2_score(actual_train_values, predicted_train_values)
        test_r2 = r2_score(actual_test_values, predicted_test_values)
        test_rmse = root_mean_squared_error(
            actual_test_values, predicted_test_values
        )
        test_r2_log = r2_score(
            np.log1p(actual_test_values), np.log1p(predicted_test_values)
        )
        train_r2_log = r2_score(
            np.log1p(actual_train_values), np.log1p(predicted_train_values)
        )

        value_min = float(min(actual_values.min(), predicted_values.min()))
        value_max = float(max(actual_values.max(), predicted_values.max()))
        value_range = value_max - value_min
        padding = 0.02 * value_range if value_range > 0 else 1.0
        axis_min = value_min - padding
        axis_max = value_max + padding

        figure, axis = plt.subplots(figsize=(6.5, 6.5), dpi=220)

        axis.scatter(
            actual_values[train_row_indices],
            predicted_values[train_row_indices],
            s=6,
            alpha=0.06,
            c="tab:blue",
            edgecolors="none",
            label="Train",
        )

        axis.scatter(
            actual_values[test_row_indices],
            predicted_values[test_row_indices],
            s=10,
            alpha=0.22,
            c="tab:orange",
            edgecolors="none",
            label="Test",
        )

        axis.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color="black",
            linewidth=1.0,
            alpha=0.8,
        )

        axis.set_title(f"{target_name}: Actual vs Predicted")
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.set_xlim(axis_min, axis_max)
        axis.set_ylim(axis_min, axis_max)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.2)

        axis.text(
            0.05,
            0.95,
            f"Train R² = {train_r2:.3f}\n"
            f"Test R² = {test_r2:.3f}\n"
            f"Test RMSE = {test_rmse:.3f}\n"
            f"R^2 log train: {train_r2_log:.3f}\n"
            f"R^2 log test: {test_r2_log:.3f}\n",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.85,
                edgecolor="none",
            ),
        )

        axis.legend(loc="lower right", frameon=True, framealpha=0.9)

        output_plot_path = os.path.join(
            plots_output_directory, f"actual_vs_predicted_{target_name}.png"
        )
        figure.tight_layout()
        figure.savefig(output_plot_path, bbox_inches="tight")
        plt.close(figure)

        log.info("Wrote plot: %s", output_plot_path)


def load_aim_csv_table(
    input_csv_path: str,
) -> tuple[pd.DataFrame, str, str, str]:
    log.info("Reading CSV: %s", input_csv_path)
    data_table = pd.read_csv(input_csv_path, sep=None, engine="python")
    log.info("Loaded %d rows, %d columns", len(data_table), data_table.shape[1])

    lower_to_original = {
        column_name.lower(): column_name for column_name in data_table.columns
    }

    latitude_column = lower_to_original.get("latitude")
    longitude_column = lower_to_original.get("longitude")
    date_column = lower_to_original.get("date") or next(
        column_name
        for column_name in data_table.columns
        if "date" in column_name.lower()
    )

    log.info(
        "Resolved columns: latitude=%s longitude=%s date=%s",
        latitude_column,
        longitude_column,
        date_column,
    )

    if latitude_column is None:
        raise KeyError("Missing required column: latitude")
    if longitude_column is None:
        raise KeyError("Missing required column: longitude")

    return data_table, latitude_column, longitude_column, date_column


def add_date_year_and_row_id_columns(
    data_table: pd.DataFrame, date_column: str
) -> pd.DataFrame:
    log.info("Parsing dates (MM/DD/YYYY) and creating year/row_id/date_str")
    data_table[date_column] = pd.to_datetime(
        data_table[date_column], format="%m/%d/%Y"
    )
    data_table["year"] = data_table[date_column].dt.year.astype(int)
    data_table["row_id"] = np.arange(len(data_table), dtype=int)
    data_table["date_str"] = data_table[date_column].dt.strftime("%Y-%m-%d")
    return data_table


def initialize_earth_engine_service_account(
    service_account_email: str, service_account_key_file: str
) -> None:
    log.info("Creating EE service account credentials")
    credentials = ee.ServiceAccountCredentials(
        service_account_email, service_account_key_file
    )
    log.info("Initializing Earth Engine")
    ee.Initialize(credentials)
    log.info("Earth Engine initialized OK")


def build_satellite_raw_band_names() -> list[str]:
    s2_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ]
    s1_bands = ["VV", "VH", "angle"]
    ls_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

    s2_indices = ["NDVI", "NDWI", "NBR", "EVI"]
    s1_indices = ["VVVH", "VHVV", "RVI"]
    ls_indices = ["NDVI", "NDWI", "NBR", "EVI"]

    band_names: list[str] = []
    band_names.extend([f"s2_{b}" for b in s2_bands])
    band_names.extend([f"s2_{i}" for i in s2_indices])
    band_names.append("s2_dt_days")

    band_names.extend([f"s1_{b}" for b in s1_bands])
    band_names.extend([f"s1_{i}" for i in s1_indices])
    band_names.append("s1_dt_days")

    band_names.extend([f"ls_{b}" for b in ls_bands])
    band_names.extend([f"ls_{i}" for i in ls_indices])
    band_names.append("ls_dt_days")
    return band_names


MISSING = -9999.0


def _filled_image(band_names: list[str], fill_value: float) -> ee.Image:
    return ee.Image.constant([fill_value] * len(band_names)).rename(band_names)


def _add_dt_bands(img: ee.Image, target_date: ee.Date) -> ee.Image:
    dt_signed = img.date().difference(target_date, "day")
    dt_abs_neg = ee.Number(dt_signed).abs().multiply(-1)

    m = img.select(0).mask()
    dt_signed_band = (
        ee.Image.constant(dt_signed).toFloat().rename("dt_signed").updateMask(m)
    )
    dt_abs_neg_band = (
        ee.Image.constant(dt_abs_neg)
        .toFloat()
        .rename("dt_abs_neg")
        .updateMask(m)
    )
    return img.addBands([dt_signed_band, dt_abs_neg_band])


def _mask_s2_sr_with_scl(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    bad = (
        scl.eq(1)
        .Or(scl.eq(3))
        .Or(scl.eq(8))
        .Or(scl.eq(9))
        .Or(scl.eq(10))
        .Or(scl.eq(11))
    )
    return image.updateMask(bad.Not())


def _add_s2_indices(img: ee.Image) -> ee.Image:
    b2 = img.select("B2")
    b4 = img.select("B4")
    b8 = img.select("B8")
    b11 = img.select("B11")
    b12 = img.select("B12")

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")
    ndwi = b8.subtract(b11).divide(b8.add(b11)).rename("NDWI")
    nbr = b8.subtract(b12).divide(b8.add(b12)).rename("NBR")
    evi = (
        b8.subtract(b4)
        .multiply(2.5)
        .divide(b8.add(b4.multiply(6.0)).subtract(b2.multiply(7.5)).add(1.0))
        .rename("EVI")
    )
    return img.addBands([ndvi, ndwi, nbr, evi])


def _build_s2_sr_closest(
    geom: ee.Geometry,
    start_date: str,
    end_date: str,
    target_date_str: str,
    max_cloud_pct: float,
) -> ee.Image:
    in_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ]
    idx_bands = ["NDVI", "NDWI", "NBR", "EVI"]
    out_bands = [f"s2_{b}" for b in (in_bands + idx_bands)]
    out_dt = "s2_dt_days"

    target_date = ee.Date(target_date_str)

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .map(_mask_s2_sr_with_scl)
        .map(
            lambda img: img.select(in_bands)
            .multiply(0.0001)
            .copyProperties(img, ["system:time_start"])
        )
        .map(_add_s2_indices)
        .map(lambda img: _add_dt_bands(img, target_date))
    )

    has = col.size().gt(0)
    composite = ee.Image(col.qualityMosaic("dt_abs_neg"))
    out = (
        composite.select(in_bands + idx_bands + ["dt_signed"])
        .rename(out_bands + [out_dt])
        .unmask(MISSING)
    )

    return ee.Image(
        ee.Algorithms.If(has, out, _filled_image(out_bands + [out_dt], MISSING))
    )


def _add_s1_indices(img: ee.Image) -> ee.Image:
    vv = img.select("VV")
    vh = img.select("VH")
    eps = ee.Image.constant(1e-6)
    vvvh = vv.divide(vh.add(eps)).rename("VVVH")
    vhvv = vh.divide(vv.add(eps)).rename("VHVV")
    rvi = vh.multiply(4.0).divide(vv.add(vh).add(eps)).rename("RVI")
    return img.addBands([vvvh, vhvv, rvi])


def _build_s1_grd_closest(
    geom: ee.Geometry,
    start_date: str,
    end_date: str,
    target_date_str: str,
) -> ee.Image:
    in_bands = ["VV", "VH", "angle"]
    idx_bands = ["VVVH", "VHVV", "RVI"]
    out_bands = [f"s1_{b}" for b in (in_bands + idx_bands)]
    out_dt = "s1_dt_days"

    target_date = ee.Date(target_date_str)

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(in_bands)
        .map(_add_s1_indices)
        .map(lambda img: _add_dt_bands(img, target_date))
    )

    has = col.size().gt(0)
    composite = ee.Image(col.qualityMosaic("dt_abs_neg"))
    out = (
        composite.select(in_bands + idx_bands + ["dt_signed"])
        .rename(out_bands + [out_dt])
        .unmask(MISSING)
    )

    return ee.Image(
        ee.Algorithms.If(has, out, _filled_image(out_bands + [out_dt], MISSING))
    )


def _mask_landsat_c2_l2(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 0)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    radsat = image.select("QA_RADSAT").eq(0)
    return image.updateMask(mask).updateMask(radsat)


def _add_ls_indices(img: ee.Image) -> ee.Image:
    b2 = img.select("SR_B2")
    b4 = img.select("SR_B4")
    b5 = img.select("SR_B5")
    b6 = img.select("SR_B6")
    b7 = img.select("SR_B7")

    ndvi = b5.subtract(b4).divide(b5.add(b4)).rename("NDVI")
    ndwi = b5.subtract(b6).divide(b5.add(b6)).rename("NDWI")
    nbr = b5.subtract(b7).divide(b5.add(b7)).rename("NBR")
    evi = (
        b5.subtract(b4)
        .multiply(2.5)
        .divide(b5.add(b4.multiply(6.0)).subtract(b2.multiply(7.5)).add(1.0))
        .rename("EVI")
    )
    return img.addBands([ndvi, ndwi, nbr, evi])


def _build_landsat_sr_closest(
    geom: ee.Geometry,
    start_date: str,
    end_date: str,
    target_date_str: str,
    max_cloud_cover: float,
) -> ee.Image:
    in_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    idx_bands = ["NDVI", "NDWI", "NBR", "EVI"]
    out_bands = [f"ls_{b}" for b in (in_bands + idx_bands)]
    out_dt = "ls_dt_days"

    target_date = ee.Date(target_date_str)

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

    col = (
        l8.merge(l9)
        .filterDate(start_date, end_date)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_cover))
        .map(_mask_landsat_c2_l2)
        .map(
            lambda img: img.select(in_bands)
            .multiply(0.0000275)
            .add(-0.2)
            .copyProperties(img, ["system:time_start"])
        )
        .map(_add_ls_indices)
        .map(lambda img: _add_dt_bands(img, target_date))
    )

    has = col.size().gt(0)
    composite = ee.Image(col.qualityMosaic("dt_abs_neg"))
    out = (
        composite.select(in_bands + idx_bands + ["dt_signed"])
        .rename(out_bands + [out_dt])
        .unmask(MISSING)
    )

    return ee.Image(
        ee.Algorithms.If(has, out, _filled_image(out_bands + [out_dt], MISSING))
    )


def _build_multisensor_image(
    geom: ee.Geometry,
    start_date: str,
    end_date: str,
    target_date_str: str,
    s2_max_cloud_pct: float,
    landsat_max_cloud_cover: float,
) -> ee.Image:
    s2 = _build_s2_sr_closest(
        geom=geom,
        start_date=start_date,
        end_date=end_date,
        target_date_str=target_date_str,
        max_cloud_pct=s2_max_cloud_pct,
    )
    s1 = _build_s1_grd_closest(
        geom=geom,
        start_date=start_date,
        end_date=end_date,
        target_date_str=target_date_str,
    )
    ls = _build_landsat_sr_closest(
        geom=geom,
        start_date=start_date,
        end_date=end_date,
        target_date_str=target_date_str,
        max_cloud_cover=landsat_max_cloud_cover,
    )
    return ee.Image.cat([s2, s1, ls]).unmask(MISSING)


def extract_satellite_raw_bands_for_points(
    point_table: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    date_column: str,
    feature_band_names: list[str],
    target_column_names: list[str],
    min_year: int,
    max_year: int,
    window_days: int,
    s2_max_cloud_pct: float,
    landsat_max_cloud_cover: float,
    chunk_size: int,
    sample_scale_meters: int,
) -> pd.DataFrame:
    property_columns = [
        "row_id",
        "date",
        "year",
        "latitude",
        "longitude",
    ] + target_column_names

    feature_chunks: list[pd.DataFrame] = []

    requested_dates = sorted(point_table["date_str"].unique().tolist())
    log.info("Extracting raw bands for %d date(s)", len(requested_dates))

    for observation_date_str in tqdm(
        requested_dates, desc="Dates", unit="date"
    ):
        observation_year = int(pd.Timestamp(observation_date_str).year)
        if not (min_year <= observation_year <= max_year):
            continue

        date_subset_table = point_table[
            point_table["date_str"] == observation_date_str
        ]
        point_count_for_date = len(date_subset_table)

        obs_date = pd.Timestamp(observation_date_str)
        start_date = (obs_date - pd.Timedelta(days=window_days)).strftime(
            "%Y-%m-%d"
        )
        end_date = (obs_date + pd.Timedelta(days=window_days + 1)).strftime(
            "%Y-%m-%d"
        )

        log.info(
            "Date %s (year=%d): %d points | window=[%s, %s)",
            observation_date_str,
            observation_year,
            point_count_for_date,
            start_date,
            end_date,
        )

        for chunk_start_index in tqdm(
            range(0, point_count_for_date, chunk_size),
            desc=f"Chunks {observation_date_str}",
            unit="chunk",
            leave=False,
        ):
            chunk_table = date_subset_table.iloc[
                chunk_start_index : chunk_start_index + chunk_size
            ]

            earth_engine_features: list[ee.Feature] = []
            for row_record in chunk_table.to_dict("records"):
                feature_properties = {
                    "row_id": int(row_record["row_id"]),
                    "date": row_record[date_column].strftime("%Y-%m-%d"),
                    "year": int(row_record["year"]),
                    "latitude": float(row_record[latitude_column]),
                    "longitude": float(row_record[longitude_column]),
                }
                for target_name in target_column_names:
                    feature_properties[target_name] = float(
                        row_record[target_name]
                    )

                point_geometry = ee.Geometry.Point(
                    [
                        feature_properties["longitude"],
                        feature_properties["latitude"],
                    ]
                )
                earth_engine_features.append(
                    ee.Feature(point_geometry, feature_properties)
                )

            feature_collection = ee.FeatureCollection(earth_engine_features)
            geom = feature_collection.geometry().bounds()

            multisensor_image = _build_multisensor_image(
                geom=geom,
                start_date=start_date,
                end_date=end_date,
                target_date_str=observation_date_str,
                s2_max_cloud_pct=s2_max_cloud_pct,
                landsat_max_cloud_cover=landsat_max_cloud_cover,
            )

            sampled_features = multisensor_image.sampleRegions(
                collection=feature_collection,
                properties=property_columns,
                scale=sample_scale_meters,
                tileScale=4,
                geometries=False,
            )

            request_start_time = time.time()
            sampled_chunk_table = ee.data.computeFeatures(
                {
                    "expression": sampled_features,
                    "fileFormat": "PANDAS_DATAFRAME",
                }
            )
            request_duration_seconds = time.time() - request_start_time

            if sampled_chunk_table is None or len(sampled_chunk_table) == 0:
                log.warning(
                    "Empty chunk result (date=%s, chunk_start=%d, chunk_len=%d) in %.2fs",
                    observation_date_str,
                    chunk_start_index,
                    len(chunk_table),
                    request_duration_seconds,
                )
                continue

            for band_name in feature_band_names:
                if band_name not in sampled_chunk_table.columns:
                    sampled_chunk_table[band_name] = np.nan
                else:
                    sampled_chunk_table[band_name] = sampled_chunk_table[
                        band_name
                    ].astype(float)
                    sampled_chunk_table.loc[
                        sampled_chunk_table[band_name] == MISSING, band_name
                    ] = np.nan

            feature_chunks.append(sampled_chunk_table)

            log.info(
                "Fetched chunk (date=%s, rows=%d, start=%d) in %.2fs",
                observation_date_str,
                len(sampled_chunk_table),
                chunk_start_index,
                request_duration_seconds,
            )

    if not feature_chunks:
        raise ValueError(
            "No samples returned from Earth Engine (all chunks empty)."
        )

    log.info("Concatenating %d chunk dataframe(s)", len(feature_chunks))
    features_table = pd.concat(feature_chunks, ignore_index=True)
    features_table["row_id"] = features_table["row_id"].astype(int)
    return features_table


def load_or_create_embeddings_table(
    point_table: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    date_column: str,
    embeddings_cache_parquet_path: str,
    feature_band_names: list[str],
    target_column_names: list[str],
    min_year: int,
    max_year: int,
    window_days: int,
    s2_max_cloud_pct: float,
    landsat_max_cloud_cover: float,
    chunk_size: int,
    sample_scale_meters: int,
) -> pd.DataFrame:
    if os.path.exists(embeddings_cache_parquet_path):
        log.info("Found cached features: %s", embeddings_cache_parquet_path)
        embeddings_table = pd.read_parquet(embeddings_cache_parquet_path)
        log.info(
            "Loaded features dataframe: %d rows, %d cols",
            len(embeddings_table),
            embeddings_table.shape[1],
        )
        for band_name in feature_band_names:
            if band_name not in embeddings_table.columns:
                embeddings_table[band_name] = np.nan
        return embeddings_table

    log.info("No cached features found; extracting from Earth Engine")
    embeddings_table = extract_satellite_raw_bands_for_points(
        point_table=point_table,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        date_column=date_column,
        feature_band_names=feature_band_names,
        target_column_names=target_column_names,
        min_year=min_year,
        max_year=max_year,
        window_days=window_days,
        s2_max_cloud_pct=s2_max_cloud_pct,
        landsat_max_cloud_cover=landsat_max_cloud_cover,
        chunk_size=chunk_size,
        sample_scale_meters=sample_scale_meters,
    )

    log.info(
        "Writing cached features parquet: %s", embeddings_cache_parquet_path
    )
    embeddings_table.to_parquet(embeddings_cache_parquet_path, index=False)
    log.info(
        "Wrote features parquet (%d rows, %d cols)",
        len(embeddings_table),
        embeddings_table.shape[1],
    )
    return embeddings_table


def train_ridge_models_and_write_metrics(
    embeddings_table: pd.DataFrame,
    embedding_band_names: list[str],
    target_column_names: list[str],
    models_output_directory: str,
    metrics_output_csv_path: str,
    ridge_alpha_grid: np.ndarray,
    train_test_seed: int,
) -> pd.DataFrame:
    os.makedirs(models_output_directory, exist_ok=True)

    log.info(
        "Building feature matrix from bands (%d bands)",
        len(embedding_band_names),
    )
    feature_matrix = embeddings_table[embedding_band_names].to_numpy(
        dtype=float
    )

    metrics_rows: list[dict] = []
    for target_name in tqdm(target_column_names, desc="Targets", unit="target"):
        log.info("Training ridge regression for target=%s", target_name)
        target_vector = embeddings_table[target_name].to_numpy(dtype=float)

        train_features, test_features, train_targets, test_targets = (
            train_test_split(
                feature_matrix,
                target_vector,
                test_size=0.2,
                random_state=train_test_seed,
            )
        )

        model_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "hgsbr",
                    HistGradientBoostingRegressor(
                        loss="poisson",
                        max_iter=12000,
                        learning_rate=0.02,
                        max_leaf_nodes=125,
                        min_samples_leaf=2,
                        l2_regularization=0.1,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=120,
                        tol=1e-7,
                        random_state=42,
                    ),
                ),
            ]
        )

        fit_start_time = time.time()
        model_pipeline.fit(train_features, train_targets)
        fit_duration_seconds = time.time() - fit_start_time

        test_predictions = model_pipeline.predict(test_features)

        test_r2 = r2_score(test_targets, test_predictions)
        test_rmse = root_mean_squared_error(test_targets, test_predictions)

        model_output_path = os.path.join(
            models_output_directory, f"ridge_{target_name}.joblib"
        )
        joblib.dump(model_pipeline, model_output_path)

        log.info(
            "Done target=%s | r2=%.4f rmse=%.4f | fit_time=%.2fs | saved=%s",
            target_name,
            test_r2,
            test_rmse,
            fit_duration_seconds,
            model_output_path,
        )

        metrics_rows.append(
            {
                "target": target_name,
                "n": int(len(target_vector)),
                "r2": float(test_r2),
                "rmse": float(test_rmse),
            }
        )

    metrics_table = pd.DataFrame(metrics_rows).sort_values("target")
    metrics_table.to_csv(metrics_output_csv_path, index=False)
    log.info("Wrote metrics: %s", metrics_output_csv_path)
    return metrics_table


def main() -> None:
    embeddings_cache_parquet_path = (
        "./data/allenai_with_satellite_raw_bands.parquet"
    )
    input_csv_path = "./data/allenai_formatted_AIM_terradata.csv"
    models_output_directory = "./models"
    metrics_output_csv_path = os.path.join(
        models_output_directory, "metrics.csv"
    )

    service_account_email = (
        "pest-control-gee@ecoshard-202922.iam.gserviceaccount.com"
    )
    service_account_key_file = "./secrets/service-account-key.json"

    feature_band_names = build_satellite_raw_band_names()

    target_column_names = [
        "Plant_sp_count",
        "BareSoilCover",
        "TotalFoliarCover",
        "FH_TotalLitterCover",
    ]

    ridge_alpha_grid = np.logspace(-3, 3, 13)
    train_test_seed = 42

    chunk_size = 500
    min_year = 2017
    max_year = 2025

    window_days = 15
    s2_max_cloud_pct = 80.0
    landsat_max_cloud_cover = 80.0
    sample_scale_meters = 30

    initialize_earth_engine_service_account(
        service_account_email=service_account_email,
        service_account_key_file=service_account_key_file,
    )

    point_table, latitude_column, longitude_column, date_column = (
        load_aim_csv_table(input_csv_path)
    )
    point_table = add_date_year_and_row_id_columns(point_table, date_column)

    embeddings_table = load_or_create_embeddings_table(
        point_table=point_table,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        date_column=date_column,
        embeddings_cache_parquet_path=embeddings_cache_parquet_path,
        feature_band_names=feature_band_names,
        target_column_names=target_column_names,
        min_year=min_year,
        max_year=max_year,
        window_days=window_days,
        s2_max_cloud_pct=s2_max_cloud_pct,
        landsat_max_cloud_cover=landsat_max_cloud_cover,
        chunk_size=chunk_size,
        sample_scale_meters=sample_scale_meters,
    )

    metrics_table = train_ridge_models_and_write_metrics(
        embeddings_table=embeddings_table,
        embedding_band_names=feature_band_names,
        target_column_names=target_column_names,
        models_output_directory=models_output_directory,
        metrics_output_csv_path=metrics_output_csv_path,
        ridge_alpha_grid=ridge_alpha_grid,
        train_test_seed=train_test_seed,
    )

    print(metrics_table.to_string(index=False))

    plots_output_directory = os.path.join(models_output_directory, "plots")
    write_actual_vs_predicted_scatterplots(
        embeddings_table=embeddings_table,
        embedding_band_names=feature_band_names,
        target_column_names=target_column_names,
        models_output_directory=models_output_directory,
        plots_output_directory=plots_output_directory,
        train_test_seed=train_test_seed,
    )


if __name__ == "__main__":
    main()
