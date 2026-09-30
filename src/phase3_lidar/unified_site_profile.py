# unified_site_profile.py
# Purpose: Merge LiDAR metrics + street-level semantics into one unified site profile

import json
from dataclasses import dataclass, asdict

from lidar_analyzer import analyze_bbox
from streetview_semantic import analyze_location


@dataclass
class EnhancedSiteProfile:
    latitude: float
    longitude: float
    avg_building_height_m: float
    canopy_cover_pct: float
    impervious_surface_pct: float
    slope_risk_score: float
    scene_composition: dict
    walkability_score: float


def build_profile(lat, lon, bbox_radius_deg=0.005):

    lidar_result = analyze_bbox(
        lat - bbox_radius_deg,
        lon - bbox_radius_deg,
        lat + bbox_radius_deg,
        lon + bbox_radius_deg
    )

    street_result = analyze_location(
        lat,
        lon
    )

    profile = EnhancedSiteProfile(
        latitude=lat,
        longitude=lon,
        avg_building_height_m=lidar_result['avg_building_height_m'],
        canopy_cover_pct=lidar_result['canopy_cover_pct'],
        impervious_surface_pct=lidar_result['impervious_surface_pct'],
        slope_risk_score=lidar_result['slope_risk_score'],
        scene_composition=street_result.get(
            'scene_composition',
            {}
        ),
        walkability_score=street_result.get(
            'walkability_score',
            0.0
        ),
    )

    return asdict(profile)


if __name__ == '__main__':

    profile = build_profile(
        13.0827,
        80.2707
    )

    print(
        json.dumps(
            profile,
            indent=2
        )
    )

    with open(
        'outputs/reports/enhanced_site_profile_example.json',
        'w'
    ) as f:

        json.dump(
            profile,
            f,
            indent=2
        )

    print(
        'Saved to outputs/reports/enhanced_site_profile_example.json'
    )