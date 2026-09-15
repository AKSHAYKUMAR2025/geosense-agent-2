import os
import pandas as pd
import joblib

from src.phase1_ml.site_scorer import score_site


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

FEATURES_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "features.csv"
)

LABELLED_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "labelled_sites.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "saved",
    "site_scorer_model.pkl"
)


def test_features_file_exists():
    assert os.path.exists(FEATURES_PATH)


def test_labelled_data_exists():
    assert os.path.exists(LABELLED_PATH)

    df = pd.read_csv(LABELLED_PATH)

    assert len(df) >= 300
    assert "label" in df.columns


def test_saved_model_exists():
    assert os.path.exists(MODEL_PATH)

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_site_scorer():
    result = score_site(13.07, 80.25)

    assert isinstance(result, dict)

    assert "latitude" in result
    assert "longitude" in result
    assert "opportunity_score" in result
    assert "risk_score" in result
    assert "features" in result
    assert "shap_explanation" in result
    assert "verdict" in result

    assert 0 <= result["opportunity_score"] <= 10
    assert 0 <= result["risk_score"] <= 10

    assert result["verdict"] in [
        "GOOD SITE",
        "POOR SITE"
    ]