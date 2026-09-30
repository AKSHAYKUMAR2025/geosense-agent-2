# lidar_features.py
# Purpose: Calculate building height, canopy cover, and slope from a classified point cloud

import laspy
import numpy as np


# Simple heuristic classifier applied on top of HeightAboveGround (HAG):
# HAG < 0.5m -> ground
# 0.5m <= HAG < 2.5m -> low vegetation / shrub
# 2.5m <= HAG < 50m -> could be tree canopy OR building
# Points with high 'return number' variability at that height -> tree
# Points with a flat, single-return roof surface -> building

def load_classified_cloud(filepath):
    las = laspy.read(filepath)

    hag = np.array(las.HeightAboveGround)
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
    num_returns = np.array(las.num_returns)
    return_num = np.array(las.return_num)

    return x, y, z, hag, num_returns, return_num


def classify_points(hag, num_returns, return_num):
    """Heuristic classification: 0=ground, 1=vegetation, 2=building."""

    labels = np.zeros(len(hag), dtype=int)

    labels[(hag >= 0.5) & (hag < 2.5)] = 1

    # Multiple returns at the same X,Y at height => light passing through leaves => tree
    tree_mask = (hag >= 2.5) & (num_returns > 1)
    labels[tree_mask] = 1

    # Single clean return at height with low return-number variance => solid roof surface
    building_mask = (hag >= 2.5) & (num_returns == 1)
    labels[building_mask] = 2

    return labels


def compute_grid_metrics(x, y, hag, labels, cell_size_m=50):
    """Aggregate metrics into a grid of cell_size_m x cell_size_m cells."""

    grid_x = ((x - x.min()) // cell_size_m).astype(int)
    grid_y = ((y - y.min()) // cell_size_m).astype(int)

    cells = {}

    for gx, gy, h, lbl in zip(grid_x, grid_y, hag, labels):
        key = (gx, gy)

        if key not in cells:
            cells[key] = {
                'building_heights': [],
                'total': 0,
                'tree_pts': 0
            }

        cells[key]['total'] += 1

        if lbl == 2:
            cells[key]['building_heights'].append(h)

        if lbl == 1:
            cells[key]['tree_pts'] += 1

    results = []

    for (gx, gy), v in cells.items():
        avg_height = (
            float(np.mean(v['building_heights']))
            if v['building_heights']
            else 0.0
        )

        canopy_pct = (
            round(100 * v['tree_pts'] / v['total'], 1)
            if v['total']
            else 0.0
        )

        results.append({
            'grid_x': gx,
            'grid_y': gy,
            'avg_building_height_m': round(avg_height, 2),
            'canopy_cover_pct': canopy_pct
        })

    return results


if __name__ == '__main__':
    x, y, z, hag, num_returns, return_num = load_classified_cloud(
        'data/lidar/processed/study_area_classified.laz'
    )

    labels = classify_points(hag, num_returns, return_num)

    metrics = compute_grid_metrics(x, y, hag, labels)

    print(f'Computed metrics for {len(metrics)} grid cells')
    print(metrics[:3])