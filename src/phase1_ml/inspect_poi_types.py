import geopandas as gpd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POI_FILE = (
    PROJECT_ROOT
    / "data"
    / "shapefiles"
    / "osm"
    / "planet_80.199,13.036_80.316,13.116-shp"
    / "shape"
    / "points.shp"
)

gdf = gpd.read_file(POI_FILE)

print("=" * 60)
print("POI TYPE INSPECTION")
print("=" * 60)

print(f"Total POIs: {len(gdf)}")
print()

print("POI types:")
print()

type_counts = gdf["type"].value_counts()

for poi_type, count in type_counts.items():
    print(f"{poi_type}: {count}")

print()
print("POI TYPE INSPECTION COMPLETE")