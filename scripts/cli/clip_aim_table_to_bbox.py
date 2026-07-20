import argparse

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter an AIM-style point table to a bounding box around its centroid."
    )
    parser.add_argument(
        "--input",
        default="data/processed/allenai_formatted_AIM_terradata.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/allenai_formatted_AIM_terradata_bbox_4x4.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--half-size-degrees",
        type=float,
        default=2.0,
        help="Half-width and half-height of the bounding box in degrees.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    lat_cols = [c for c in df.columns if c.lower() in ("latitude", "lat")]
    lon_cols = [
        c
        for c in df.columns
        if c.lower() in ("longitude", "lon", "lng", "long")
    ]

    lat_col = lat_cols[0]
    lon_col = lon_cols[0]

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    centroid_lat = df[lat_col].mean(skipna=True)
    centroid_lon = df[lon_col].mean(skipna=True)

    min_lat = centroid_lat - args.half_size_degrees
    max_lat = centroid_lat + args.half_size_degrees
    min_lon = centroid_lon - args.half_size_degrees
    max_lon = centroid_lon + args.half_size_degrees

    filtered = df[
        df[lat_col].between(min_lat, max_lat, inclusive="both")
        & df[lon_col].between(min_lon, max_lon, inclusive="both")
    ].copy()

    filtered.to_csv(args.output, index=False)
    print(f"wrote {len(filtered)} of {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
