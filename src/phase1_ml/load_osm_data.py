import os
from pathlib import Path

import geopandas as gpd
from dotenv import load_dotenv
from sqlalchemy import create_engine


# --------------------------------------------------
# 1. Load database settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# --------------------------------------------------
# 2. OSM data location
# --------------------------------------------------

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "shapefiles"
    / "osm"
    / "planet_80.199,13.036_80.316,13.116-shp"
    / "shape"
)


# --------------------------------------------------
# 3. OSM layers
# --------------------------------------------------

LAYERS = [
    ("roads.shp", "osm_roads"),
    ("buildings.shp", "osm_buildings"),
    ("landuse.shp", "osm_landuse"),
    ("points.shp", "osm_poi"),
]


# --------------------------------------------------
# 4. Load each layer
# --------------------------------------------------

for filename, table_name in LAYERS:

    filepath = DATA_DIR / filename

    print("=" * 60)
    print(f"Loading: {filename}")
    print(f"PostGIS table: {table_name}")
    print("=" * 60)

    gdf = gpd.read_file(filepath)

    print(f"Features read: {len(gdf)}")
    print(f"CRS: {gdf.crs}")

    # Make sure coordinates use WGS84 latitude/longitude
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)

    print(f"Writing to PostgreSQL table: {table_name}")

    gdf.to_postgis(
        table_name,
        engine,
        if_exists="replace",
        index=False
    )

    print(f"SUCCESS: {table_name} loaded")
    print()


print("=" * 60)
print("ALL OSM LAYERS LOADED SUCCESSFULLY")
print("=" * 60)