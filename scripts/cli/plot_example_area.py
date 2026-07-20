import argparse

import matplotlib.pyplot as plt
from shapely import wkt


DEFAULT_WKT = (
    "POLYGON ((-99.24852828966696 36.99699001819784, "
    "-99.24852828966696 48.993367778284366, "
    "-113.67844756796137 48.993367778284366, "
    "-113.67844756796137 36.99699001819784, "
    "-99.24852828966696 36.99699001819784))"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a WKT polygon in EPSG:4326 coordinates."
    )
    parser.add_argument("--wkt", default=DEFAULT_WKT, help="Input polygon WKT.")
    parser.add_argument(
        "--output",
        default="outputs/figures/example-area.png",
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
    poly = wkt.loads(args.wkt)
    x, y = poly.exterior.xy

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, color="crimson", linewidth=2)
    ax.fill(x, y, color="crimson", alpha=0.2)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Polygon (EPSG:4326)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", dpi=220)
    print(f"wrote {args.output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
