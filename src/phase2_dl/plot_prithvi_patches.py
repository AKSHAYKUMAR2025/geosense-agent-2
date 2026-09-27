from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import contextily as ctx


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_prithvi_patch_mapping.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "prithvi_patch_map_satellite.png"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Geographic raster extent
# ---------------------------------------------------------

LEFT = 80.19889362033852
RIGHT = 80.31603393338771
BOTTOM = 13.035992077028848
TOP = 13.116121800372309

WIDTH = 652
HEIGHT = 446

PATCH_SIZE = 224


# ---------------------------------------------------------
# Convert WGS84 longitude/latitude to Web Mercator
# ---------------------------------------------------------

def lon_to_x(lon):
    return (
        lon
        * 20037508.34
        / 180.0
    )


def lat_to_y(lat):
    lat = max(
        min(lat, 85.05112878),
        -85.05112878,
    )

    lat_rad = math.radians(lat)

    return (
        math.log(
            math.tan(
                math.pi / 4
                + lat_rad / 2
            )
        )
        * 20037508.34
        / math.pi
    )


# ---------------------------------------------------------
# Load candidates
# ---------------------------------------------------------

df = pd.read_csv(
    CANDIDATES_FILE
)

print(
    "Candidates:",
    len(df),
)


# ---------------------------------------------------------
# Convert candidate coordinates
# ---------------------------------------------------------

df["x"] = df["longitude"].apply(
    lon_to_x
)

df["y"] = df["latitude"].apply(
    lat_to_y
)


# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 10)
)


# ---------------------------------------------------------
# Calculate geographic plot bounds
# ---------------------------------------------------------

x_left_scene = lon_to_x(LEFT)
x_right_scene = lon_to_x(RIGHT)

y_bottom_scene = lat_to_y(BOTTOM)
y_top_scene = lat_to_y(TOP)

padding_x = (
    x_right_scene - x_left_scene
) * 0.02

padding_y = (
    y_top_scene - y_bottom_scene
) * 0.02


ax.set_xlim(
    x_left_scene - padding_x,
    x_right_scene + padding_x,
)

ax.set_ylim(
    y_bottom_scene - padding_y,
    y_top_scene + padding_y,
)


# ---------------------------------------------------------
# Add Esri satellite imagery
# ---------------------------------------------------------

ctx.add_basemap(
    ax,
    source=ctx.providers.Esri.WorldImagery,
    zoom=13,
)


# ---------------------------------------------------------
# Draw six Prithvi patches
# ---------------------------------------------------------

for patch_id in range(6):

    row = patch_id // 3
    col = patch_id % 3

    x_pixel = col * PATCH_SIZE
    y_pixel = row * PATCH_SIZE

    # Pixel → longitude
    lon_left = (
        LEFT
        + (x_pixel / WIDTH)
        * (RIGHT - LEFT)
    )

    lon_right = (
        LEFT
        + (
            min(
                x_pixel + PATCH_SIZE,
                WIDTH,
            )
            / WIDTH
        )
        * (RIGHT - LEFT)
    )

    # Pixel → latitude
    lat_top = (
        TOP
        - (y_pixel / HEIGHT)
        * (TOP - BOTTOM)
    )

    lat_bottom = (
        TOP
        - (
            min(
                y_pixel + PATCH_SIZE,
                HEIGHT,
            )
            / HEIGHT
        )
        * (TOP - BOTTOM)
    )

    # Convert patch corners to Web Mercator
    x1 = lon_to_x(lon_left)
    x2 = lon_to_x(lon_right)

    y1 = lat_to_y(lat_bottom)
    y2 = lat_to_y(lat_top)

    rectangle = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor="white",
        linewidth=2.5,
    )

    ax.add_patch(
        rectangle
    )

    # Patch label
    ax.text(
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        f"Patch {patch_id}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.65,
            pad=4,
        ),
    )


# ---------------------------------------------------------
# Plot candidate locations
# ---------------------------------------------------------

for patch_id in sorted(
    df["patch_id"].unique()
):

    subset = df[
        df["patch_id"] == patch_id
    ]

    ax.scatter(
        subset["x"],
        subset["y"],
        s=24,
        label=(
            f"Patch {patch_id} "
            f"({len(subset)})"
        ),
        edgecolors="white",
        linewidths=0.4,
    )


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

ax.set_title(
    "GeoSense Phase 2 — Prithvi Patch Coverage",
    fontsize=17,
    fontweight="bold",
)

ax.set_xlabel(
    "Web Mercator X",
    fontsize=11,
)

ax.set_ylabel(
    "Web Mercator Y",
    fontsize=11,
)

ax.legend(
    title="Prithvi patches",
    loc="upper right",
    framealpha=0.9,
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=250,
    bbox_inches="tight",
)

plt.close()

print()
print("Satellite basemap map saved:")
print(OUTPUT_FILE)

print()
print(
    "PRITHVI SATELLITE PATCH MAP COMPLETE"
)