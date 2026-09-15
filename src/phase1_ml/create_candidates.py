import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point


# --------------------------------------------------
# Project location
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------
# Chennai study area
# --------------------------------------------------

STUDY_AREA_BBOX = (
    80.199,  # minimum longitude
    13.036,  # minimum latitude
    80.316,  # maximum longitude
    13.116   # maximum latitude
)

GRID_SPACING_DEGREES = 0.005


# --------------------------------------------------
# Create candidate grid
# --------------------------------------------------

min_lon, min_lat, max_lon, max_lat = STUDY_AREA_BBOX

lons = np.arange(
    min_lon,
    max_lon,
    GRID_SPACING_DEGREES
)

lats = np.arange(
    min_lat,
    max_lat,
    GRID_SPACING_DEGREES
)

points = [
    Point(lon, lat)
    for lon in lons
    for lat in lats
]


gdf = gpd.GeoDataFrame(
    {
        "location_id": range(len(points)),
        "longitude": [point.x for point in points],
        "latitude": [point.y for point in points],
        "geometry": points
    },
    crs="EPSG:4326"
)


print("=" * 60)
print("CANDIDATE GRID CREATED")
print("=" * 60)

print(f"Number of candidate locations: {len(gdf)}")
print(f"CRS: {gdf.crs}")


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "candidate_locations.shp"

gdf.to_file(OUTPUT_FILE)

print()
print(f"Saved to:")
print(OUTPUT_FILE)
print()
print("CANDIDATE GRID COMPLETE")