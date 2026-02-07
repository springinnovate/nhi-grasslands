import logging
import os
import time

import ee
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
)
log = logging.getLogger(__name__)


def load_aim_csv_table(
    input_csv_path: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Loads the AIM-style CSV and resolves coordinate and date columns.

    Args:
        input_csv_path: Path to the input CSV. Must contain 'latitude' and 'longitude'
            columns, and a date column named 'date' or containing 'date' in its name.

    Returns:
        A tuple of:
            - data_table: Loaded DataFrame.
            - latitude_column: Column name for latitude.
            - longitude_column: Column name for longitude.
            - date_column: Column name for date.

    Raises:
        KeyError: If required columns are missing.
        StopIteration: If no date-like column is found.
    """
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
    data_table: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """Parses the date column and adds 'year' and 'row_id' columns.

    Args:
        data_table: Input DataFrame.
        date_column: Name of the date column with format MM/DD/YYYY.

    Returns:
        The same DataFrame with columns:
            - year: integer year parsed from date_column
            - row_id: unique integer row identifier

    Raises:
        ValueError: If date parsing fails due to unexpected format.
    """
    log.info("Parsing dates (MM/DD/YYYY) and creating year/row_id")
    data_table[date_column] = pd.to_datetime(
        data_table[date_column], format="%m/%d/%Y"
    )
    data_table["year"] = data_table[date_column].dt.year.astype(int)
    data_table["row_id"] = np.arange(len(data_table), dtype=int)
    return data_table


def initialize_earth_engine_service_account(
    service_account_email: str,
    service_account_key_file: str,
) -> None:
    """Initializes the Earth Engine client using a service account key.

    Args:
        service_account_email: Service account email address.
        service_account_key_file: Path to the JSON service account key file.

    Raises:
        ee.EEException: If initialization fails.
    """
    log.info("Creating EE service account credentials")
    credentials = ee.ServiceAccountCredentials(
        service_account_email, service_account_key_file
    )

    log.info("Initializing Earth Engine")
    ee.Initialize(credentials)
    log.info("Earth Engine initialized OK")


def build_satellite_embedding_band_names(
    embedding_dimension_count: int,
) -> list[str]:
    """Builds Earth Engine band names for the Satellite Embedding dataset.

    Args:
        embedding_dimension_count: Number of embedding dimensions.

    Returns:
        A list of band names: ['A00', 'A01', ..., 'A{N-1:02d}'].
    """
    return [
        f"A{band_index:02d}" for band_index in range(embedding_dimension_count)
    ]


def extract_satellite_embeddings_for_points(
    point_table: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    date_column: str,
    embedding_imagecollection_id: str,
    embedding_band_names: list[str],
    target_column_names: list[str],
    min_embedding_year: int,
    max_embedding_year: int,
    chunk_size: int,
) -> pd.DataFrame:
    """Extracts annual satellite embeddings for point observations.

    For each year in the input table within [min_embedding_year, max_embedding_year],
    samples the annual embedding mosaic at each point.

    Args:
        point_table: DataFrame containing columns: row_id, year, latitude, longitude, and targets.
        latitude_column: Name of latitude column.
        longitude_column: Name of longitude column.
        date_column: Name of date column (datetime64 after parsing).
        embedding_imagecollection_id: Earth Engine ImageCollection ID for embeddings.
        embedding_band_names: Band names to select (e.g., A00..A63).
        target_column_names: Target columns to include as properties in the EE FeatureCollection.
        min_embedding_year: Minimum year to process.
        max_embedding_year: Maximum year to process.
        chunk_size: Number of points to sample per request.

    Returns:
        DataFrame containing original properties plus embedding bands as columns.

    Raises:
        ee.EEException: If Earth Engine sampling fails.
    """
    embedding_collection = ee.ImageCollection(embedding_imagecollection_id)

    property_columns = [
        "row_id",
        "date",
        "year",
        "latitude",
        "longitude",
    ] + target_column_names

    embedding_chunks: list[pd.DataFrame] = []

    requested_years = sorted(point_table["year"].unique().tolist())
    log.info(
        "Extracting embeddings for %d year(s): %s",
        len(requested_years),
        requested_years,
    )

    for observation_year in tqdm(requested_years, desc="Years", unit="year"):
        if not (min_embedding_year <= observation_year <= max_embedding_year):
            continue

        year_subset_table = point_table[point_table["year"] == observation_year]
        year_start_date = f"{observation_year}-01-01"
        year_end_date = f"{observation_year + 1}-01-01"
        year_embedding_collection = embedding_collection.filterDate(
            year_start_date, year_end_date
        )

        point_count_for_year = len(year_subset_table)
        log.info("Year %d: %d points", observation_year, point_count_for_year)

        for chunk_start_index in tqdm(
            range(0, point_count_for_year, chunk_size),
            desc=f"Chunks {observation_year}",
            unit="chunk",
            leave=False,
        ):
            chunk_table = year_subset_table.iloc[
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

            embedding_image = (
                year_embedding_collection.filterBounds(
                    feature_collection.geometry()
                )
                .mosaic()
                .select(embedding_band_names)
            )

            sampled_features = embedding_image.sampleRegions(
                collection=feature_collection,
                properties=property_columns,
                scale=10,
                tileScale=4,
                geometries=False,
            )

            request_start_time = time.time()
            try:
                sampled_chunk_table = ee.data.computeFeatures(
                    {
                        "expression": sampled_features,
                        "fileFormat": "PANDAS_DATAFRAME",
                    }
                )
            except Exception:
                log.exception(
                    "computeFeatures failed (year=%d, chunk_start=%d, chunk_len=%d)",
                    observation_year,
                    chunk_start_index,
                    len(chunk_table),
                )
                raise
            request_duration_seconds = time.time() - request_start_time

            if sampled_chunk_table is None or len(sampled_chunk_table) == 0:
                log.warning(
                    "Empty chunk result (year=%d, chunk_start=%d, chunk_len=%d) in %.2fs",
                    observation_year,
                    chunk_start_index,
                    len(chunk_table),
                    request_duration_seconds,
                )
            else:
                log.info(
                    "Fetched chunk (year=%d, rows=%d, start=%d) in %.2fs",
                    observation_year,
                    len(sampled_chunk_table),
                    chunk_start_index,
                    request_duration_seconds,
                )

            embedding_chunks.append(sampled_chunk_table)

    log.info("Concatenating %d chunk dataframe(s)", len(embedding_chunks))
    embeddings_table = pd.concat(embedding_chunks, ignore_index=True)
    embeddings_table["row_id"] = embeddings_table["row_id"].astype(int)
    return embeddings_table


def load_or_create_embeddings_table(
    point_table: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    date_column: str,
    embeddings_cache_parquet_path: str,
    embedding_imagecollection_id: str,
    embedding_band_names: list[str],
    target_column_names: list[str],
    min_embedding_year: int,
    max_embedding_year: int,
    chunk_size: int,
) -> pd.DataFrame:
    """Loads cached embeddings from Parquet or generates them via Earth Engine.

    Args:
        point_table: DataFrame of point observations with year/row_id columns.
        latitude_column: Latitude column name.
        longitude_column: Longitude column name.
        date_column: Date column name.
        embeddings_cache_parquet_path: Path to Parquet cache file.
        embedding_imagecollection_id: Earth Engine ImageCollection ID.
        embedding_band_names: Embedding band names to sample.
        target_column_names: Target columns to carry through.
        min_embedding_year: Minimum year to process.
        max_embedding_year: Maximum year to process.
        chunk_size: Chunk size for sampling.

    Returns:
        DataFrame with embeddings and targets.
    """
    if os.path.exists(embeddings_cache_parquet_path):
        log.info("Found cached embeddings: %s", embeddings_cache_parquet_path)
        embeddings_table = pd.read_parquet(embeddings_cache_parquet_path)
        log.info(
            "Loaded embeddings dataframe: %d rows, %d cols",
            len(embeddings_table),
            embeddings_table.shape[1],
        )
        return embeddings_table

    log.info("No cached embeddings found; extracting from Earth Engine")
    embeddings_table = extract_satellite_embeddings_for_points(
        point_table=point_table,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        date_column=date_column,
        embedding_imagecollection_id=embedding_imagecollection_id,
        embedding_band_names=embedding_band_names,
        target_column_names=target_column_names,
        min_embedding_year=min_embedding_year,
        max_embedding_year=max_embedding_year,
        chunk_size=chunk_size,
    )

    log.info(
        "Writing cached embeddings parquet: %s", embeddings_cache_parquet_path
    )
    embeddings_table.to_parquet(embeddings_cache_parquet_path, index=False)
    log.info(
        "Wrote embeddings parquet (%d rows, %d cols)",
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
    """Trains one ridge regression per target and writes models + metrics to disk.

    Args:
        embeddings_table: DataFrame containing embedding bands and targets.
        embedding_band_names: Names of embedding band columns.
        target_column_names: Target columns to predict.
        models_output_directory: Directory to write serialized models.
        metrics_output_csv_path: Path to write metrics CSV.
        ridge_alpha_grid: Candidate ridge alphas for RidgeCV.
        train_test_seed: Random seed for train/test split.

    Returns:
        A DataFrame of metrics per target.
    """
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
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=ridge_alpha_grid, cv=5)),
            ]
        )

        fit_start_time = time.time()
        model_pipeline.fit(train_features, train_targets)
        fit_duration_seconds = time.time() - fit_start_time

        test_predictions = model_pipeline.predict(test_features)

        test_r2 = r2_score(test_targets, test_predictions)
        test_rmse = root_mean_squared_error(test_targets, test_predictions)
        selected_alpha = float(model_pipeline.named_steps["ridge"].alpha_)

        model_output_path = os.path.join(
            models_output_directory, f"ridge_{target_name}.joblib"
        )
        joblib.dump(model_pipeline, model_output_path)

        log.info(
            "Done target=%s | r2=%.4f rmse=%.4f alpha=%s | fit_time=%.2fs | saved=%s",
            target_name,
            test_r2,
            test_rmse,
            selected_alpha,
            fit_duration_seconds,
            model_output_path,
        )

        metrics_rows.append(
            {
                "target": target_name,
                "n": int(len(target_vector)),
                "r2": float(test_r2),
                "rmse": float(test_rmse),
                "alpha": selected_alpha,
            }
        )

    metrics_table = pd.DataFrame(metrics_rows).sort_values("target")
    metrics_table.to_csv(metrics_output_csv_path, index=False)
    log.info("Wrote metrics: %s", metrics_output_csv_path)
    return metrics_table


def main() -> None:
    embeddings_cache_parquet_path = (
        "./data/allenai_with_satellite_embeddings.parquet"
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

    embedding_imagecollection_id = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    embedding_dimension_count = 64
    embedding_band_names = build_satellite_embedding_band_names(
        embedding_dimension_count
    )

    target_column_names = [
        "Plant_sp_count",
        "BareSoilCover",
        "TotalFoliarCover",
        "FH_TotalLitterCover",
    ]

    ridge_alpha_grid = np.logspace(-3, 3, 13)
    train_test_seed = 42

    chunk_size = 500
    min_embedding_year = 2017
    max_embedding_year = 2025

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
        embedding_imagecollection_id=embedding_imagecollection_id,
        embedding_band_names=embedding_band_names,
        target_column_names=target_column_names,
        min_embedding_year=min_embedding_year,
        max_embedding_year=max_embedding_year,
        chunk_size=chunk_size,
    )

    metrics_table = train_ridge_models_and_write_metrics(
        embeddings_table=embeddings_table,
        embedding_band_names=embedding_band_names,
        target_column_names=target_column_names,
        models_output_directory=models_output_directory,
        metrics_output_csv_path=metrics_output_csv_path,
        ridge_alpha_grid=ridge_alpha_grid,
        train_test_seed=train_test_seed,
    )

    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
