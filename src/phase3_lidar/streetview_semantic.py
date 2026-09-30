# streetview_semantic.py
# PURPOSE: Classify 360-degree street imagery into scene categories and compute a walkability score
# INPUT: latitude (float), longitude (float)
# OUTPUT: dict with scene composition percentages and a walkability score

import os
import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from mapillary_download import find_images_in_bbox, download_image


MODEL_NAME = 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'


# Cityscapes label ids we care about for a walkability score
CLASS_NAMES = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    8: 'vegetation',
    10: 'sky',
    13: 'car'
}


_processor = None
_model = None


def _load_model():
    global _processor, _model

    if _model is None:
        _processor = SegformerImageProcessor.from_pretrained(
            MODEL_NAME
        )

        _model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME
        )

        _model.eval()

    return _processor, _model


def classify_image(image_path):
    """Returns pixel-percentage breakdown of classes for one street image."""

    processor, model = _load_model()

    image = Image.open(image_path).convert('RGB')

    inputs = processor(
        images=image,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)

    seg = outputs.logits.argmax(
        dim=1
    )[0].numpy()

    total_pixels = seg.size

    composition = {}

    for class_id, name in CLASS_NAMES.items():
        pct = round(
            100 * np.sum(seg == class_id) / total_pixels,
            1
        )

        composition[name] = pct

    return composition


def compute_walkability_score(composition):
    """Higher sidewalk + vegetation, lower car dominance = more walkable. Scale 0-10."""

    score = (
        composition.get('sidewalk', 0) * 0.4
        + composition.get('vegetation', 0) * 0.3
        - composition.get('car', 0) * 0.2
    )

    return round(
        max(0, min(10, score / 5)),
        2
    )


def analyze_location(lat, lon, radius_deg=0.0015):
    """
    Main function: analyse street-level scene composition near a location.

    Returns: dict with per-class percentages (averaged across images)
    and walkability_score
    """

    images = find_images_in_bbox(
        lon - radius_deg,
        lat - radius_deg,
        lon + radius_deg,
        lat + radius_deg,
        limit=10
    )

    if not images:
        return {
            'latitude': lat,
            'longitude': lon,
            'error': 'No street imagery found nearby'
        }

    all_compositions = []

    for img_meta in images:
        path = download_image(img_meta)

        all_compositions.append(
            classify_image(path)
        )

    avg_composition = {
        cls: round(
            float(
                np.mean(
                    [c[cls] for c in all_compositions]
                )
            ),
            1
        )
        for cls in CLASS_NAMES.values()
    }

    return {
        'latitude': lat,
        'longitude': lon,
        'images_analyzed': len(all_compositions),
        'scene_composition': avg_composition,
        'walkability_score': compute_walkability_score(
            avg_composition
        )
    }


if __name__ == '__main__':
    result = analyze_location(
        13.05215569,
        80.24310426
    )

    print(result)