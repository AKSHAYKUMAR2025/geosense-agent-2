# pdal_pipeline.py
# Purpose: Clean a raw LAZ point cloud — remove noise, classify ground vs non-ground

import pdal
import json
import os


def build_pipeline(input_laz, output_laz):
    """Returns a PDAL pipeline definition as a Python dict."""
    pipeline_def = [
        input_laz,
        {
            'type': 'filters.outlier',
            'method': 'statistical',
            'mean_k': 8,
            'multiplier': 2.5,
        },
        {
            # Classifies points into ground (2) vs non-ground using a
            # Simple Morphological Filter — the PDAL standard approach
            'type': 'filters.smrf',
        },
        {
            # Adds a 'HeightAboveGround' dimension
            'type': 'filters.hag_nn',
        },
        {
            'type': 'writers.las',
            'filename': output_laz,
            'extra_dims': 'all',
        },
    ]
    return pipeline_def


def run_pipeline(input_laz, output_laz):
    pipeline_def = build_pipeline(input_laz, output_laz)
    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    count = pipeline.execute()
    print(f'Processed {count} points. Output saved to {output_laz}')
    return pipeline


if __name__ == '__main__':
    os.makedirs('data/lidar/processed', exist_ok=True)

    run_pipeline(
        'data/lidar/raw/20181029_Faraglione_for_OT.laz',
        'data/lidar/processed/study_area_classified.laz'
    )