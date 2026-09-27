from pathlib import Path

import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_locations.shp"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_prithvi_patch_mapping.csv"
)


# ---------------------------------------------------------
# Raster / patch information
# ---------------------------------------------------------

RASTER_LEFT = 80.19889362033852
RASTER_RIGHT = 80.31603393338771
RASTER_BOTTOM = 13.035992077028848
RASTER_TOP = 13.116121800372309

RASTER_WIDTH = 652
RASTER_HEIGHT = 446

PATCH_SIZE = 224

# Six patches:
# 0 1 2
# 3 4 5


# ---------------------------------------------------------
# Load candidates
# ---------------------------------------------------------

gdf = gpd.read_file(CANDIDATES_FILE)

print("Candidates loaded:", len(gdf))


# ---------------------------------------------------------
# Convert geographic coordinates to raster pixels
# ---------------------------------------------------------

def longitude_to_pixel(lon):
    return (
        (lon - RASTER_LEFT)
        / (RASTER_RIGHT - RASTER_LEFT)
        * RASTER_WIDTH
    )


def latitude_to_pixel(lat):
    # Raster row 0 is at the TOP.
    return (
        (RASTER_TOP - lat)
        / (RASTER_TOP - RASTER_BOTTOM)
        * RASTER_HEIGHT
    )


gdf["pixel_x"] = gdf["longitude"].apply(longitude_to_pixel)
gdf["pixel_y"] = gdf["latitude"].apply(latitude_to_pixel)


# ---------------------------------------------------------
# Determine patch
# ---------------------------------------------------------

gdf["patch_col"] = (
    gdf["pixel_x"] // PATCH_SIZE
).astype(int)

gdf["patch_row"] = (
    gdf["pixel_y"] // PATCH_SIZE
).astype(int)


# Safety check
gdf["patch_col"] = gdf["patch_col"].clip(0, 2)
gdf["patch_row"] = gdf["patch_row"].clip(0, 1)


gdf["patch_id"] = (
    gdf["patch_row"] * 3
    + gdf["patch_col"]
)


# ---------------------------------------------------------
# Keep useful columns
# ---------------------------------------------------------

result = gdf[
    [
        "location_i",
        "longitude",
        "latitude",
        "pixel_x",
        "pixel_y",
        "patch_row",
        "patch_col",
        "patch_id",
    ]
].copy()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print()
print("Patch assignment:")
print(
    result["patch_id"]
    .value_counts()
    .sort_index()
)

print()
print("Expected patch IDs: 0, 1, 2, 3, 4, 5")

print()
print("Unique patches used:")
print(
    sorted(result["patch_id"].unique())
)

print()
print("Candidates mapped:", len(result))

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("CANDIDATE → PRITHVI PATCH MAPPING COMPLETE")