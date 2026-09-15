import geopandas as gpd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "shapefiles"
    / "osm"
    / "planet_80.199,13.036_80.316,13.116-shp"
    / "shape"
)

files = {
    "roads": DATA_DIR / "roads.shp",
    "buildings": DATA_DIR / "buildings.shp",
    "landuse": DATA_DIR / "landuse.shp",
    "points": DATA_DIR / "points.shp",
}

for name, filepath in files.items():
    print("=" * 60)
    print(f"{name.upper()}")
    print("=" * 60)

    gdf = gpd.read_file(filepath)

    print("Number of features:", len(gdf))
    print("Columns:")

    for column in gdf.columns:
        print("  -", column)

    print()

print("OSM ATTRIBUTE INSPECTION COMPLETE")