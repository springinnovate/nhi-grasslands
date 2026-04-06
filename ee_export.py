from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
import json
import re

import ee


SKIP_STATES = {"READY", "RUNNING", "COMPLETED", "CANCEL_REQUESTED"}
DEFAULT_EXPORT_KWARGS = {
    "maxPixels": 1e13,
    "shardSize": 512,
    "fileDimensions": 16384,
    "skipEmptyTiles": True,
    "fileFormat": "GeoTIFF",
}
TASK_NAMESPACE = "gee"


def init_ee():
    key_file = "/workdir/secrets/service-account-key.json"
    service_account = json.load(open(key_file))["client_email"]
    credentials = ee.ServiceAccountCredentials(service_account, key_file)
    ee.Initialize(credentials)


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


@dataclass(frozen=True)
class ExportWindow:
    label: str
    start: str = None
    end: str = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExportPlan:
    bucket: str
    description: str
    search_descriptions: tuple
    file_name_prefix: str
    build_image: object
    export_kwargs: dict
    region_factory: object


def year_windows(start_year, end_year):
    for year in range(start_year, end_year + 1):
        yield ExportWindow(
            label=f"year_{year}",
            start=f"{year}-01-01",
            end=f"{year + 1}-01-01",
            meta={"year": year},
        )


def month_windows(start_year, end_year):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = date(year, month, 1)
            end_date = date(
                year + (month == 12), 1 if month == 12 else month + 1, 1
            )
            yield ExportWindow(
                label=f"month_{year}_{month:02d}",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                meta={"year": year, "month": month},
            )


def day_windows(start_date, end_date):
    current = date.fromisoformat(start_date)
    stop = date.fromisoformat(end_date)
    while current <= stop:
        next_day = current + timedelta(days=1)
        yield ExportWindow(
            label=f"day_{current:%Y_%m_%d}",
            start=current.isoformat(),
            end=next_day.isoformat(),
            meta={
                "year": current.year,
                "month": current.month,
                "day": current.day,
            },
        )
        current = next_day


def month_windows_range(start_date, end_date):
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    current = current.replace(day=1)

    while current <= end:
        next_month = date(
            current.year + (current.month == 12),
            1 if current.month == 12 else current.month + 1,
            1,
        )

        yield ExportWindow(
            label=f"month_{current.year}_{current.month:02d}",
            start=current.isoformat(),
            end=next_month.isoformat(),
            meta={"year": current.year, "month": current.month},
        )

        current = next_month


def stack_images(images):
    image = ee.Image(images[0])
    for next_image in images[1:]:
        image = image.addBands(next_image)
    return image


def build_task_index():
    task_states = defaultdict(set)
    for task in ee.batch.Task.list():
        status = task.status()
        description = status.get("description")
        state = status.get("state")
        if description and state:
            task_states[description].add(state)
    return task_states


def blocking_matches(task_states, descriptions):
    matches = {}
    for description in descriptions:
        states = sorted(
            state
            for state in task_states.get(description, set())
            if state in SKIP_STATES
        )
        if states:
            matches[description] = states
    return matches


class LayerExportPlugin:
    layer_name = ""
    version = "v1"
    bucket = "ecoshard-root"
    export_root = "gee_export"
    extra_export_kwargs = {}
    force_download = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def iter_windows(self):
        yield ExportWindow(label="all")

    def build_image(self, window):
        raise NotImplementedError

    def region(self, image, window):
        return image.geometry()

    def description(self, window):
        return "__".join(
            [TASK_NAMESPACE, slug(self.layer_name), self.version, window.label]
        )

    def aliases(self, window):
        return ()

    def file_name_prefix(self, window):
        return f"{self.export_root}/{slug(self.layer_name)}/{self.version}/{window.label}"

    def export_kwargs(self, window):
        return {**DEFAULT_EXPORT_KWARGS, **self.extra_export_kwargs}

    def plans(self):
        for window in self.iter_windows():
            description = self.description(window)
            yield ExportPlan(
                bucket=self.bucket,
                description=description,
                search_descriptions=(description, *self.aliases(window)),
                file_name_prefix=self.file_name_prefix(window),
                build_image=lambda window=window: self.build_image(window),
                export_kwargs=self.export_kwargs(window),
                region_factory=lambda image, window=window: self.region(
                    image, window
                ),
            )


class GrasslandProbabilityLayer(LayerExportPlugin):
    layer_name = "nat_semi_grassland_p"
    version = "v1"
    collection_id = (
        "projects/global-pasture-watch/assets/ggc-30m/v1/nat-semi-grassland_p"
    )

    def iter_windows(self):
        return year_windows(2000, 2022)

    def aliases(self, window):
        year = window.meta["year"]
        return (f"nat_semi_grassland_p_{year}",)

    def build_image(self, window):
        return ee.Image(
            ee.ImageCollection(self.collection_id)
            .filterDate(window.start, window.end)
            .first()
        )

    def region(self, image, window):
        return ee.Geometry.Rectangle([-180, -90, 180, 90], geodesic=False)


class HumanImpactIndex(LayerExportPlugin):
    layer_name = "hii"
    version = "v1"
    collection_id = "projects/HII"

    def iter_windows(self):
        return year_windows(2000, 2022)

    def aliases(self, window):
        year = window.meta["year"]
        return f"hii_{year}"

    def build_image(self, window):
        return ee.Image(
            ee.ImageCollection(self.collection_id)
            .filterDate(window.start, window.end)
            .first()
        )

    def region(self, image, window):
        return ee.Geometry.Rectangle([-180, -90, 180, 90], geodesic=False)


class AnnualDominantClassofGrasslands(LayerExportPlugin):
    layer_name = "grassland_c"
    version = "v1"
    collection_id = (
        "projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c"
    )

    def iter_windows(self):
        return year_windows(2012, 2012)

    def aliases(self, window):
        year = window.meta["year"]
        return f"annual_dominant_grassland_class_{year}"

    def build_image(self, window):
        return ee.Image(
            ee.ImageCollection(self.collection_id)
            .filterDate(window.start, window.end)
            .first()
        )

    def region(self, image, window):
        return ee.Geometry.Rectangle([-180, -90, 180, 90], geodesic=False)


class Era5MonthlyTemperatureLayer(LayerExportPlugin):
    layer_name = "era5_t2m_monthly"
    version = "v1"
    collection_id = "ECMWF/ERA5/MONTHLY"

    era5_bands = [
        "mean_2m_air_temperature",
        "minimum_2m_air_temperature",
        "maximum_2m_air_temperature",
        "dewpoint_2m_temperature",
        "total_precipitation",
        "surface_pressure",
        "mean_sea_level_pressure",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
    ]

    def iter_windows(self):
        return month_windows_range("1979-03-01", "2020-06-01")

    def build_image(self, window):
        year = window.meta["year"]
        month = window.meta["month"]

        return (
            ee.ImageCollection(self.collection_id)
            .filterDate(window.start, window.end)
            .first()
            .select(self.era5_bands)
            .set("band_order", self.era5_bands)
            .set("year", year)
            .set("month", month)
            .set("window_start", window.start)
            .set("window_end", window.end)
        )

    def aliases(self, window):
        year = window.meta["year"]
        month = window.meta["month"]
        return (f"era5_monthly_{year}_{month:02d}",)

    def region(self, image, window):
        global_bounds = ee.Geometry.Rectangle(
            [-180, -90, 180, 90],
            proj="EPSG:4326",
            geodesic=False,
        )
        return global_bounds


def run_export_layers(layers):
    task_states = build_task_index()
    started = 0
    skipped = 0

    for layer in layers:
        for plan in layer.plans():
            matches = blocking_matches(task_states, plan.search_descriptions)
            if matches and not layer.force_download:
                print(
                    f"skipping {plan.description} because existing task(s) found: {matches}"
                )
                skipped += 1
                continue

            image = plan.build_image()
            task = ee.batch.Export.image.toCloudStorage(
                image=image,
                description=plan.description,
                bucket=plan.bucket,
                fileNamePrefix=plan.file_name_prefix,
                region=plan.region_factory(image),
                **plan.export_kwargs,
            )
            task.start()
            task_states[plan.description].add("READY")
            print(f"started {plan.description}")
            started += 1

    print(f"done started={started} skipped={skipped}")


def main():
    init_ee()

    layers = [
        # Era5MonthlyTemperatureLayer(),
        # AnnualDominantClassofGrasslands(),
    ]

    run_export_layers(layers)


if __name__ == "__main__":
    main()
