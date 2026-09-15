import os
import joblib
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "saved",
    "site_scorer_model.pkl"
)

FEATURES_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "features.csv"
)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Load feature data
# ---------------------------------------------------------

features_df = pd.read_csv(FEATURES_PATH)

FEATURE_COLUMNS = [
    "dist_road_m",
    "dist_hospital_m",
    "flood_risk"
]


# ---------------------------------------------------------
# SHAP explainer
# ---------------------------------------------------------

explainer = shap.TreeExplainer(model)


# ---------------------------------------------------------
# Find nearest candidate location
# ---------------------------------------------------------

def find_nearest_location(latitude, longitude):
    distances = (
        (features_df["latitude"] - latitude) ** 2
        + (features_df["longitude"] - longitude) ** 2
    )

    nearest_index = distances.idxmin()

    return features_df.loc[nearest_index]


# ---------------------------------------------------------
# Score a site
# ---------------------------------------------------------

def score_site(latitude, longitude):

    site = find_nearest_location(latitude, longitude)

    X = pd.DataFrame(
        [[
            site["dist_road_m"],
            site["dist_hospital_m"],
            site["flood_risk"]
        ]],
        columns=FEATURE_COLUMNS
    )

    # Model probability
    probabilities = model.predict_proba(X)[0]

    opportunity_score = float(probabilities[1] * 10)

    risk_score = float(probabilities[0] * 10)

    # SHAP explanation
    shap_values = explainer.shap_values(X)

    # SHAP 0.51+ may return an array with shape:
    # (samples, features, classes)
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        shap_array = np.asarray(shap_values)

        if shap_array.ndim == 3:
            # Class 1 = GOOD SITE
            values = shap_array[0, :, 1]
        elif shap_array.ndim == 2:
            values = shap_array[0]
        else:
            values = shap_array

    shap_explanation = {
        feature: float(value)
        for feature, value in zip(FEATURE_COLUMNS, values)
    }
    # Verdict
    verdict = "GOOD SITE" if opportunity_score > 5 else "POOR SITE"

    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "opportunity_score": round(opportunity_score, 2),
        "risk_score": round(risk_score, 2),
        "features": {
            "dist_road_m": round(float(site["dist_road_m"]), 2),
            "dist_hospital_m": round(float(site["dist_hospital_m"]), 2),
            "flood_risk": round(float(site["flood_risk"]), 2)
        },
        "shap_explanation": shap_explanation,
        "verdict": verdict
    }


# ---------------------------------------------------------
# Test from command line
# ---------------------------------------------------------

if __name__ == "__main__":

    test_latitude = 13.07
    test_longitude = 80.25

    result = score_site(
        test_latitude,
        test_longitude
    )

    print("\n" + "=" * 60)
    print("GEOSENSE SITE SCORER")
    print("=" * 60)

    print("\nLatitude:", result["latitude"])
    print("Longitude:", result["longitude"])

    print("\nOpportunity Score:",
          result["opportunity_score"], "/ 10")

    print("Risk Score:",
          result["risk_score"], "/ 10")

    print("\nFeatures:")

    for feature, value in result["features"].items():
        print(f"  {feature}: {value}")

    print("\nSHAP Explanation:")

    for feature, value in result["shap_explanation"].items():
        print(f"  {feature}: {value:.4f}")

    print("\nVerdict:", result["verdict"])

    print("=" * 60)