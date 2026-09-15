import pandas as pd
import joblib
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import xgboost as xgb
import shap
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labelled_sites.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

FEATURES = [
    "dist_road_m",
    "dist_hospital_m",
    "flood_risk"
]

TARGET = "label"


print("=" * 60)
print("GEOSENSE ML MODEL TRAINING")
print("=" * 60)


# ---------------------------------------------------------
# Load labelled data
# ---------------------------------------------------------

print()
print("Loading labelled data...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} labelled locations.")


# ---------------------------------------------------------
# Prepare X and y
# ---------------------------------------------------------

X = df[FEATURES]
y = df[TARGET]

print()
print("Features:")
print(FEATURES)

print()
print("Label distribution:")
print(y.value_counts())


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

print()
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_score = rf.score(X_test, y_test)

print(f"Random Forest Accuracy: {rf_score:.3f}")

print()
print("Random Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf.predict(X_test)
    )
)


# ---------------------------------------------------------
# XGBoost
# ---------------------------------------------------------

print()
print("Training XGBoost...")

xg = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric="logloss"
)

xg.fit(X_train, y_train)

xg_score = xg.score(X_test, y_test)

print(f"XGBoost Accuracy: {xg_score:.3f}")

print()
print("XGBoost Classification Report:")
print(
    classification_report(
        y_test,
        xg.predict(X_test)
    )
)


# ---------------------------------------------------------
# Select best model
# ---------------------------------------------------------

if rf_score >= xg_score:

    best = rf
    best_name = "RandomForest"

else:

    best = xg
    best_name = "XGBoost"


print()
print("=" * 60)
print(f"BEST MODEL: {best_name}")
print("=" * 60)


# ---------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------

print()
print("Calculating SHAP feature importance...")

explainer = shap.TreeExplainer(best)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=FEATURES,
    show=False
)


# ---------------------------------------------------------
# Save SHAP plot
# ---------------------------------------------------------

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

shap_file = REPORT_DIR / "shap_importance.png"

plt.savefig(
    shap_file,
    bbox_inches="tight"
)

plt.close()

print()
print("SHAP plot saved to:")
print(shap_file)


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model_file = MODEL_DIR / "site_scorer_model.pkl"

joblib.dump(
    best,
    model_file
)

print()
print("Best model saved to:")
print(model_file)


print()
print("=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)