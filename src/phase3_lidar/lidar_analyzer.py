# lidar_analyzer.py
# PURPOSE: The main LiDAR analysis module for GeoSense Agent
# INPUT: bounding box (min_lat, min_lon, max_lat, max_lon)
# OUTPUT: dict with avg building height, tree canopy %, impervious surface %, slope risk score

import numpy as np

from lidar_features import (
    load_classified_cloud,
    classify_points,
    compute_grid_metrics
)


PROCESSED_CLOUD_PATH = 'data/lidar/processed/study_area_classified.laz'


def _compute_slope_risk(z_values):
    """A simple proxy: higher elevation variance = steeper terrain = higher slope risk (0-10)."""

    if len(z_values) < 2:
        return 0.0

    std = float(np.std(z_values))

    return round(
        min(10.0, std * 2),
        2
    )


def analyze_bbox(min_lat, min_lon, max_lat, max_lon):
    """
    Main function: analyse LiDAR-derived metrics for a bounding box.

    Returns: dict with avg_building_height_m,
    canopy_cover_pct, impervious_surface_pct,
    slope_risk_score
    """

    x, y, z, hag, num_returns, return_num = load_classified_cloud(
        PROCESSED_CLOUD_PATH
    )

    labels = classify_points(
        hag,
        num_returns,
        return_num
    )

    # NOTE: for a production version, filter x/y here to only the requested lat/lon bbox
    # (reprojected to the point cloud's coordinate system) before aggregating.

    building_heights = hag[labels == 2]

    tree_pct = (
        round(
            100 * np.sum(labels == 1) / len(labels),
            1
        )
        if len(labels)
        else 0.0
    )

    building_pct = (
        round(
            100 * np.sum(labels == 2) / len(labels),
            1
        )
        if len(labels)
        else 0.0
    )

    avg_height = (
        float(np.mean(building_heights))
        if len(building_heights)
        else 0.0
    )

    return {
        'bbox': [
            min_lat,
            min_lon,
            max_lat,
            max_lon
        ],
        'avg_building_height_m': round(
            avg_height,
            2
        ),
        'canopy_cover_pct': tree_pct,
        'impervious_surface_pct': building_pct,
        'slope_risk_score': _compute_slope_risk(z),
        'total_points_analyzed': int(len(hag)),
    }


if __name__ == '__main__':

    result = analyze_bbox(
        13.05,
        80.24,
        13.06,
        80.25
    )

    for key, value in result.items():
        print(f'{key:24s}: {value}')