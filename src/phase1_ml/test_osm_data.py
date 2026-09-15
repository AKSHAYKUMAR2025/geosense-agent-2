import geopandas as gpd
from pathlib import Path

# Find the GeoSense project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "shapefiles"
    / "osm"
    / "planet_80.199,13.036_80.316,13.116-shp"
    / "shape"
)

files_to_test = {
    "roads": DATA_DIR / "roads.shp",
    "buildings": DATA_DIR / "buildings.shp",
    "landuse": DATA_DIR / "landuse.shp",
    "points": DATA_DIR / "points.shp",
}

print("GeoSense project root:")
print(PROJECT_ROOT)
print()

print("OSM data folder:")
print(DATA_DIR)
print()

print("Checking OSM shapefiles...")
print()

for name, filepath in files_to_test.items():

    print(f"Reading {name}:")
    print(f"  File: {filepath}")

    if not filepath.exists():
        print("  ERROR: File not found")
        print()
        continue

    gdf = gpd.read_file(filepath)

    print(f"  Features: {len(gdf)}")
    print(f"  CRS: {gdf.crs}")
    print(
        f"  Geometry types: "
        f"{gdf.geometry.geom_type.value_counts().to_dict()}"
    )
    print("  OK")
    print()

print("OSM DATA CHECK COMPLETE")