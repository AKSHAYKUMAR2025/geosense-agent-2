import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

from db_connection import get_engine


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calc_distance_to_nearest(candidates_gdf, layer_table, engine):
    """
    Calculate distance in metres from each candidate point
    to the nearest feature in a PostGIS table.
    """

    # Convert candidate points to metre-based CRS
    candidates_projected = candidates_gdf.to_crs(epsg=32644)

    # Read geometry from PostGIS
    layer = gpd.read_postgis(
        f"SELECT geometry FROM {layer_table}",
        engine,
        geom_col="geometry"
    )

    # Convert OSM layer to metre-based CRS
    layer_projected = layer.to_crs(epsg=32644)

    # Calculate nearest distance
    distances = candidates_projected.geometry.apply(
        lambda point: layer_projected.geometry.distance(point).min()
    )

    return distances


def calc_distance_to_hospital(candidates_gdf, engine):
    """
    Calculate distance in metres to the nearest hospital.
    Only POIs whose type is 'hospital' are used.
    """

    candidates_projected = candidates_gdf.to_crs(epsg=32644)

    hospital_query = """
        SELECT geometry
        FROM osm_poi
        WHERE type = 'hospital'
    """

    hospitals = gpd.read_postgis(
        hospital_query,
        engine,
        geom_col="geometry"
    )

    hospitals_projected = hospitals.to_crs(epsg=32644)

    distances = candidates_projected.geometry.apply(
        lambda point: hospitals_projected.geometry.distance(point).min()
    )

    return distances


def assign_flood_risk(candidates_gdf, engine):
    """
    Assign flood risk score.

    high   = 8
    medium = 5
    low    = 2
    no flood-zone match = 0

    If flood_zones does not exist yet, use 0 as the
    temporary fallback, following the project manual.
    """

    query = """
        SELECT geometry, risk_level
        FROM flood_zones
    """

    try:
        flood_gdf = gpd.read_postgis(
            query,
            engine,
            geom_col="geometry"
        )

        joined = gpd.sjoin(
            candidates_gdf,
            flood_gdf,
            how="left",
            predicate="within"
        )

        risk_map = {
            "high": 8,
            "medium": 5,
            "low": 2
        }

        return joined["risk_level"].map(risk_map).fillna(0).reset_index(drop=True)

    except Exception:
        print("Flood zones table not found - using temporary risk value 0")

        return pd.Series(
            np.zeros(len(candidates_gdf))
        )


def build_feature_table(candidates_gdf, engine):

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Make sure required candidate columns exist
    if "location_id" not in candidates_gdf.columns:
        candidates_gdf = candidates_gdf.copy()
        candidates_gdf["location_id"] = range(len(candidates_gdf))

    if "latitude" not in candidates_gdf.columns:
        candidates_gdf["latitude"] = candidates_gdf.geometry.y

    if "longitude" not in candidates_gdf.columns:
        candidates_gdf["longitude"] = candidates_gdf.geometry.x

    df = candidates_gdf[
        ["location_id", "latitude", "longitude"]
    ].copy()

    print()
    print("Calculating distance to nearest road...")

    df["dist_road_m"] = calc_distance_to_nearest(
        candidates_gdf,
        "osm_roads",
        engine
    )

    print("Road distance calculated.")

    print()
    print("Calculating distance to nearest hospital...")

    df["dist_hospital_m"] = calc_distance_to_hospital(
        candidates_gdf,
        engine
    )

    print("Hospital distance calculated.")

    print()
    print("Calculating flood risk...")

    df["flood_risk"] = assign_flood_risk(
        candidates_gdf,
        engine
    )

    print("Flood risk calculated.")

    print()
    print(
        f"Feature table built with {len(df)} locations "
        f"and {len(df.columns)} columns"
    )

    print()
    print("Feature columns:")
    print(df.columns.tolist())

    print()
    print("First 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":

    print("Connecting to GeoSense database...")

    engine = get_engine()

    candidate_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "candidate_locations.shp"
    )

    print("Loading candidate locations...")
    print(candidate_file)

    candidates = gpd.read_file(candidate_file)

    print(f"Loaded {len(candidates)} candidate locations.")

    features = build_feature_table(
        candidates,
        engine
    )

    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "features.csv"
    )

    features.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print(f"Features saved to:")
    print(output_file)