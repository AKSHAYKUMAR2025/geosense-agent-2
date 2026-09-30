# test_phase3_modules.py
import sys
sys.path.append('src/phase3_lidar')

from lidar_analyzer import analyze_bbox
from unified_site_profile import build_profile


def test_lidar_analyzer_returns_dict():
    result = analyze_bbox(13.05, 80.24, 13.06, 80.25)
    assert isinstance(result, dict), 'Result should be a dictionary'


def test_lidar_analyzer_required_keys():
    result = analyze_bbox(13.05, 80.24, 13.06, 80.25)
    required = [
        'avg_building_height_m',
        'canopy_cover_pct',
        'impervious_surface_pct',
        'slope_risk_score'
    ]

    for key in required:
        assert key in result, f'Missing key: {key}'


def test_canopy_cover_in_valid_range():
    result = analyze_bbox(13.05, 80.24, 13.06, 80.25)
    assert 0 <= result['canopy_cover_pct'] <= 100, \
        'Canopy cover must be 0-100'


def test_unified_profile_has_all_fields():
    profile = build_profile(13.0827, 80.2707)
    required = [
        'avg_building_height_m',
        'canopy_cover_pct',
        'scene_composition',
        'walkability_score'
    ]

    for key in required:
        assert key in profile, f'Missing key: {key}'