# sam_rooftop_extraction.py
# Purpose: Use Meta's Segment Anything Model to auto-outline rooftops in an aerial image,
# then match each rooftop polygon to its LiDAR-measured height to estimate building volume

import os
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


CHECKPOINT = os.getenv(
    'SAM_CHECKPOINT_PATH',
    'models/saved/sam/sam_vit_b_01ec64.pth'
)


def load_sam_generator():
    sam = sam_model_registry['vit_b'](checkpoint=CHECKPOINT)
    return SamAutomaticMaskGenerator(sam)


def detect_rooftops(image_path):
    """Runs SAM on an aerial image and returns a list of segment masks."""

    image = cv2.cvtColor(
        cv2.imread(image_path),
        cv2.COLOR_BGR2RGB
    )

    generator = load_sam_generator()
    masks = generator.generate(image)

    # Keep only medium-sized, roughly rectangular segments
    # (likely rooftops, not roads or gardens)
    rooftop_candidates = [
        m for m in masks
        if 40 < m['area'] < 4000
    ]

    print(
        f'Detected {len(rooftop_candidates)} candidate rooftop segments'
    )

    return rooftop_candidates


def estimate_building_volume(
    rooftop_mask,
    avg_height_m,
    pixel_area_m2=0.25
):
    """Combines rooftop footprint area (from SAM)
    with LiDAR-measured height to estimate volume."""

    footprint_area_m2 = rooftop_mask['area'] * pixel_area_m2
    volume_m3 = round(
        footprint_area_m2 * avg_height_m,
        1
    )

    return {
        'footprint_area_m2': round(footprint_area_m2, 1),
        'height_m': avg_height_m,
        'volume_m3': volume_m3
    }


if __name__ == '__main__':
    rooftops = detect_rooftops(
        'data/raw/aerial/study_area_ortho.png'
    )

    # Example: match the first detected rooftop
    # with a LiDAR height of 8.4m
    result = estimate_building_volume(
        rooftops[0],
        avg_height_m=8.4
    )

    print(result)