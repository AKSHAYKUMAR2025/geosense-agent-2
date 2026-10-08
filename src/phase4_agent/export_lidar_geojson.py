import json
from pathlib import Path

import laspy
import requests
import numpy as np

from pyproj import Transformer
from shapely.geometry import Polygon

try:
    from shapely import contains_xy
except ImportError:
    contains_xy = None


# =========================================================
# GeoSense Agent
# OSM Building Footprints + Phase 3 LiDAR Heights
# =========================================================


# =========================================================
# 1. PATHS
# =========================================================

# Current script location:
#
# GeoSense_Agent
# └── src
#     └── scripts
#         └── export_osm_lidar_buildings.py
#
# parents[2] = GeoSense_Agent

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LIDAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "lidar"
    / "processed"
    / "study_area_classified.laz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "frontend"
    / "digital_twin"
    / "public"
    / "lidar_buildings.geojson"
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

OVERPASS_URL = (
    "https://overpass-api.de/api/interpreter"
)

LIDAR_EPSG = 32633
WGS84_EPSG = 4326

# Minimum LiDAR building points required
# inside an OSM footprint.
MIN_LIDAR_POINTS = 10


# =========================================================
# 3. START
# =========================================================

print("========================================")
print("GeoSense OSM + LiDAR Building Export")
print("========================================")

print("\nProject root:")
print(PROJECT_ROOT)

print("\nLiDAR input:")
print(LIDAR_PATH)


if not LIDAR_PATH.exists():
    raise FileNotFoundError(
        f"LiDAR file not found:\n{LIDAR_PATH}"
    )


# =========================================================
# 4. READ LIDAR
# =========================================================

print("\nReading LiDAR file...")

las = laspy.read(LIDAR_PATH)

print(
    f"LiDAR points: {len(las.points):,}"
)

print(
    f"LiDAR CRS: EPSG:{LIDAR_EPSG}"
)


# =========================================================
# 5. LOAD PHASE 3 VARIABLES
# =========================================================

print("\nReading HeightAboveGround...")

hag = np.asarray(
    las.HeightAboveGround,
    dtype=float,
)

num_returns = np.asarray(
    las.num_returns,
)

x = np.asarray(
    las.x,
    dtype=float,
)

y = np.asarray(
    las.y,
    dtype=float,
)


print(
    "HeightAboveGround points:",
    f"{len(hag):,}"
)


# =========================================================
# 6. APPLY THE PHASE 3 BUILDING CLASSIFIER
# =========================================================
#
# This follows the Phase 3 manual:
#
# HAG < 0.5 m
#     -> ground
#
# 0.5 <= HAG < 2.5 m
#     -> low vegetation
#
# HAG >= 2.5 m AND num_returns > 1
#     -> vegetation/tree
#
# HAG >= 2.5 m AND num_returns == 1
#     -> building
#
# We only need the building mask here.
#


print(
    "\nApplying Phase 3 building classification..."
)


building_mask = (
    (hag >= 2.5)
    & (num_returns == 1)
)


building_x = x[building_mask]
building_y = y[building_mask]
building_hag = hag[building_mask]


print(
    "Phase 3 building-classified points:",
    f"{len(building_x):,}"
)


if len(building_x) == 0:
    raise RuntimeError(
        "The Phase 3 building classifier produced "
        "zero building points."
    )


# =========================================================
# 7. GET LIDAR BOUNDS
# =========================================================

min_x = float(np.min(x))
max_x = float(np.max(x))

min_y = float(np.min(y))
max_y = float(np.max(y))


print("\nLiDAR UTM bounds:")

print(
    f"  X: {min_x:.2f} -> {max_x:.2f}"
)

print(
    f"  Y: {min_y:.2f} -> {max_y:.2f}"
)


# =========================================================
# 8. CONVERT BOUNDS TO WGS84
# =========================================================

print(
    "\nConverting LiDAR bounds to WGS84..."
)


transform_to_wgs84 = Transformer.from_crs(
    LIDAR_EPSG,
    WGS84_EPSG,
    always_xy=True,
)


west, south = transform_to_wgs84.transform(
    min_x,
    min_y,
)

east, north = transform_to_wgs84.transform(
    max_x,
    max_y,
)


print("\nLiDAR WGS84 bounds:")

print(
    f"  Longitude: {west:.6f} -> {east:.6f}"
)

print(
    f"  Latitude:  {south:.6f} -> {north:.6f}"
)


# =========================================================
# 9. QUERY OPENSTREETMAP
# =========================================================

print(
    "\nQuerying OpenStreetMap building footprints..."
)


query = f"""
[out:json][timeout:120];
way[building](
    {south},
    {west},
    {north},
    {east}
);
out geom;
"""


response = requests.post(
    OVERPASS_URL,
    data=query,
    headers={
        "User-Agent": "GeoSense-Agent/2.0"
    },
    timeout=180,
)


print(
    "Overpass HTTP status:",
    response.status_code,
)


response.raise_for_status()

osm_data = response.json()

elements = osm_data.get(
    "elements",
    [],
)


print(
    "OSM building ways returned:",
    len(elements),
)


# =========================================================
# 10. BUILD OSM POLYGONS
# =========================================================

print(
    "\nConverting OSM buildings to polygons..."
)


osm_polygons = []

for element in elements:

    geometry = element.get(
        "geometry"
    )

    if not geometry:
        continue

    if len(geometry) < 3:
        continue

    coordinates = [
        (
            point["lon"],
            point["lat"],
        )
        for point in geometry
    ]

    polygon = Polygon(
        coordinates
    )

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if polygon.is_empty:
        continue

    if polygon.geom_type != "Polygon":
        continue

    osm_polygons.append(
        {
            "osm_id": int(
                element["id"]
            ),
            "polygon_wgs84": polygon,
        }
    )


print(
    "Valid OSM building polygons:",
    len(osm_polygons),
)


if not osm_polygons:
    raise RuntimeError(
        "No valid OSM building polygons "
        "were found."
    )


# =========================================================
# 11. TRANSFORM OSM POLYGONS TO UTM
# =========================================================

print(
    "\nProjecting OSM footprints to LiDAR CRS..."
)


transform_to_utm = Transformer.from_crs(
    WGS84_EPSG,
    LIDAR_EPSG,
    always_xy=True,
)


for item in osm_polygons:

    polygon_wgs84 = item[
        "polygon_wgs84"
    ]

    projected_coordinates = [
        transform_to_utm.transform(
            lon,
            lat,
        )
        for lon, lat
        in polygon_wgs84.exterior.coords
    ]

    polygon_utm = Polygon(
        projected_coordinates
    )

    if not polygon_utm.is_valid:
        polygon_utm = polygon_utm.buffer(0)

    item[
        "polygon_utm"
    ] = polygon_utm


# =========================================================
# 12. MATCH LIDAR BUILDING POINTS
#     TO OSM FOOTPRINTS
# =========================================================

print(
    "\nMatching Phase 3 LiDAR building points "
    "to OSM footprints..."
)


features = []

matched_buildings = 0

rejected_buildings = 0


for index, item in enumerate(
    osm_polygons,
    start=1,
):

    osm_id = item[
        "osm_id"
    ]

    polygon_utm = item[
        "polygon_utm"
    ]

    print(
        f"  Processing building "
        f"{index}/{len(osm_polygons)} "
        f"(OSM {osm_id})..."
    )


    # -----------------------------------------------------
    # Polygon bounding box
    # -----------------------------------------------------

    bounds = polygon_utm.bounds

    polygon_min_x = bounds[0]
    polygon_min_y = bounds[1]
    polygon_max_x = bounds[2]
    polygon_max_y = bounds[3]


    # -----------------------------------------------------
    # First filter points using NumPy.
    #
    # This avoids checking every LiDAR point against
    # every building polygon.
    # -----------------------------------------------------

    bbox_mask = (
        (building_x >= polygon_min_x)
        & (building_x <= polygon_max_x)
        & (building_y >= polygon_min_y)
        & (building_y <= polygon_max_y)
    )


    candidate_x = building_x[
        bbox_mask
    ]

    candidate_y = building_y[
        bbox_mask
    ]

    candidate_hag = building_hag[
        bbox_mask
    ]


    if len(candidate_x) == 0:

        rejected_buildings += 1

        continue


    # -----------------------------------------------------
    # Exact point-in-polygon test
    # -----------------------------------------------------

    if contains_xy is not None:

        inside_mask = contains_xy(
            polygon_utm,
            candidate_x,
            candidate_y,
        )

    else:

        # Compatibility fallback for older Shapely.
        inside_mask = np.array(
            [
                polygon_utm.contains(
                    __import__(
                        "shapely.geometry",
                        fromlist=["Point"],
                    ).Point(
                        px,
                        py,
                    )
                )
                for px, py in zip(
                    candidate_x,
                    candidate_y,
                )
            ]
        )


    matched_hag = candidate_hag[
        inside_mask
    ]


    # -----------------------------------------------------
    # Require enough LiDAR evidence
    # -----------------------------------------------------

    if len(matched_hag) < MIN_LIDAR_POINTS:

        rejected_buildings += 1

        continue


    # -----------------------------------------------------
    # Calculate LiDAR height
    # -----------------------------------------------------
    #
    # Phase 3 uses the mean HAG for building points
    # when calculating average building height.
    #
    # We also calculate robust percentiles for reporting.
    # -----------------------------------------------------

    mean_height = float(
        np.mean(matched_hag)
    )

    p10_height = float(
        np.percentile(
            matched_hag,
            10,
        )
    )

    p95_height = float(
        np.percentile(
            matched_hag,
            95,
        )
    )


    # -----------------------------------------------------
    # Reject clearly invalid heights
    # -----------------------------------------------------

    if mean_height < 2.0:

        rejected_buildings += 1

        continue


    if mean_height > 100.0:

        rejected_buildings += 1

        continue


    # -----------------------------------------------------
    # Calculate footprint area
    # -----------------------------------------------------

    area_m2 = float(
        polygon_utm.area
    )


    # -----------------------------------------------------
    # Convert polygon back to WGS84 coordinates
    # for GeoJSON.
    # -----------------------------------------------------

    polygon_wgs84 = item[
        "polygon_wgs84"
    ]


    coordinates = [
        [
            float(lon),
            float(lat),
        ]
        for lon, lat
        in polygon_wgs84.exterior.coords
    ]


    # -----------------------------------------------------
    # Create GeoJSON feature
    # -----------------------------------------------------

    feature = {
        "type": "Feature",

        "properties": {

            "osm_id": osm_id,

            "height_m": round(
                mean_height,
                2,
            ),

            "height_p10_m": round(
                p10_height,
                2,
            ),

            "height_p95_m": round(
                p95_height,
                2,
            ),

            "area_m2": round(
                area_m2,
                2,
            ),

            "lidar_points": int(
                len(matched_hag)
            ),

            "source": (
                "OpenStreetMap building footprint "
                "+ Phase 3 LiDAR height"
            ),

            "classification_rule": (
                "HAG >= 2.5m and "
                "num_returns == 1"
            ),
        },

        "geometry": {

            "type": "Polygon",

            "coordinates": [
                coordinates
            ],
        },
    }


    features.append(
        feature
    )

    matched_buildings += 1


# =========================================================
# 13. CREATE GEOJSON
# =========================================================

geojson = {
    "type": "FeatureCollection",

    "features": features,
}


# =========================================================
# 14. SAVE OUTPUT
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        geojson,
        f,
        indent=2,
    )


# =========================================================
# 15. FINAL SUMMARY
# =========================================================

print("\n========================================")
print("EXPORT COMPLETE")
print("========================================")

print(
    "\nOSM buildings found:",
    len(osm_polygons),
)

print(
    "Buildings matched with LiDAR:",
    matched_buildings,
)

print(
    "Buildings rejected:",
    rejected_buildings,
)

print(
    "\nOutput file:"
)

print(
    OUTPUT_PATH
)


# =========================================================
# 16. HEIGHT SUMMARY
# =========================================================

if features:

    heights = [
        feature[
            "properties"
        ]["height_m"]
        for feature in features
    ]


    print(
        "\nLiDAR-derived building heights:"
    )

    print(
        f"  Minimum: "
        f"{min(heights):.2f} m"
    )

    print(
        f"  Maximum: "
        f"{max(heights):.2f} m"
    )

    print(
        f"  Average: "
        f"{np.mean(heights):.2f} m"
    )


    lidar_counts = [
        feature[
            "properties"
        ]["lidar_points"]
        for feature in features
    ]


    print(
        "\nLiDAR points per matched building:"
    )

    print(
        f"  Minimum: "
        f"{min(lidar_counts)}"
    )

    print(
        f"  Maximum: "
        f"{max(lidar_counts)}"
    )

    print(
        f"  Average: "
        f"{np.mean(lidar_counts):.1f}"
    )


else:

    print(
        "\nWARNING:"
    )

    print(
        "No OSM buildings had sufficient "
        "LiDAR building points."
    )