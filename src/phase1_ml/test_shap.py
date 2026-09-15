import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labelled_sites.csv"
)


print("Loading data...")

df = pd.read_csv(INPUT_FILE)

FEATURES = [
    "dist_road_m",
    "dist_hospital_m",
    "flood_risk"
]

X = df[FEATURES]
y = df["label"]


print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

print("Random Forest ready.")

print("Creating SHAP explainer...")

explainer = shap.Explainer(
    rf,
    X_train
)

print("Calculating SHAP values...")

shap_values = explainer(X_test)

print("SHAP calculation successful!")

print("SHAP values shape:", shap_values.values.shape)

print("TEST COMPLETE")