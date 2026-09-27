from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prithvi_features.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_FILE = (
    MODEL_DIR
    / "combined_phase1_prithvi_classifier.pkl"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(FEATURE_FILE)

print("=" * 60)
print("GeoSense Phase 2 - Phase 1 + Prithvi Classifier")
print("=" * 60)

print()
print("Rows:", len(df))


# ---------------------------------------------------------
# Phase 1 features
# ---------------------------------------------------------

phase1_candidates = [
    "dist_road_m",
    "dist_hospital_m",
    "flood_risk",
]

missing_phase1 = [
    column
    for column in phase1_candidates
    if column not in df.columns
]

if missing_phase1:
    raise ValueError(
        f"Missing Phase 1 features: {missing_phase1}"
    )


# ---------------------------------------------------------
# Prithvi features
# ---------------------------------------------------------

prithvi_columns = [
    column
    for column in df.columns
    if column.startswith("prithvi_")
]

print("Phase 1 features:", phase1_candidates)
print("Prithvi features:", len(prithvi_columns))


# ---------------------------------------------------------
# Build combined feature matrix
# ---------------------------------------------------------

feature_columns = (
    phase1_candidates
    + prithvi_columns
)

X = df[feature_columns].to_numpy(
    dtype=np.float32
)

y = df["label"].to_numpy(
    dtype=np.int64
)


print("Total features:", X.shape[1])


# ---------------------------------------------------------
# Same split as Prithvi-only experiment
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print()
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

print()
print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print()
print("=" * 60)
print("COMBINED CLASSIFIER RESULTS")
print("=" * 60)

print()
print("Accuracy:", accuracy)

print()
print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)

print()
print("Classification report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
    )
)


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE,
)

print()
print("Model saved:")
print(MODEL_FILE)

print()
print("COMBINED CLASSIFIER COMPLETE")