# visualize_cloud.py
# Purpose: Colour-code the classified point cloud and save it as a viewable PLY file

import open3d as o3d
import numpy as np

from lidar_features import load_classified_cloud, classify_points


COLOR_MAP = {
    0: [0.55, 0.35, 0.15],
    1: [0.1, 0.6, 0.1],
    2: [0.6, 0.6, 0.6]
}  # ground=brown, tree=green, building=grey


def save_colored_ply(input_laz, output_ply):
    x, y, z, hag, num_returns, return_num = load_classified_cloud(input_laz)

    labels = classify_points(hag, num_returns, return_num)

    points = np.vstack((x, y, z)).T
    colors = np.array([COLOR_MAP[l] for l in labels])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_ply, pcd)

    print(f'Saved colour-coded point cloud to {output_ply}')


if __name__ == '__main__':
    save_colored_ply(
        'data/lidar/processed/study_area_classified.laz',
        'outputs/point_clouds/study_area_colored.ply'
    )