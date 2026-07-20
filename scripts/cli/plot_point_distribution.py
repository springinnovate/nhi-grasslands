import argparse

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the latitude/longitude distribution of AIM points."
    )
    parser.add_argument(
        "--points",
        default="data/raw/AIM_TerraDat_4RS_wCDL(in).csv",
        help="Input point CSV.",
    )
    parser.add_argument(
        "--countries",
        default="data/external/countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg",
        help="Country boundary GeoPackage.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/point-distribution.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively after writing it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.points)
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    countries = gpd.read_file(args.countries)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    countries = countries.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(14, 7))

    countries.plot(
        ax=ax, facecolor="#f5f5f5", edgecolor="#777777", linewidth=0.5
    )
    gdf_points.plot(ax=ax, markersize=3, alpha=0.35, color="#1f77b4")

    xmin, ymin, xmax, ymax = gdf_points.total_bounds
    pad_x = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
    pad_y = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Point Distribution (Latitude/Longitude)")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", dpi=220)
    print(f"wrote {args.output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
