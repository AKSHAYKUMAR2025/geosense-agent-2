from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "phase2_cross_validation.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(FEATURE_FILE)

y = df["label"].to_numpy(dtype=np.int64)


# ---------------------------------------------------------
# Feature groups
# ---------------------------------------------------------

phase1_features = [
    "dist_road_m",
    "dist_hospital_m",
    "flood_risk",
]

prithvi_features = [
    column
    for column in df.columns
    if column.startswith("prithvi_")
]


# ---------------------------------------------------------
# Cross-validation setup
# ---------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ---------------------------------------------------------
# Model factory
# ---------------------------------------------------------

def create_model():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


# ---------------------------------------------------------
# Experiments
# ---------------------------------------------------------

experiments = {
    "Phase 1 baseline": phase1_features,

    "Prithvi only": prithvi_features,

    "Phase 1 + Prithvi": (
        phase1_features
        + prithvi_features
    ),
}


results = []


print("=" * 60)
print("GeoSense Phase 2 - 5-Fold Cross-Validation")
print("=" * 60)


# ---------------------------------------------------------
# Run experiments
# ---------------------------------------------------------

for name, features in experiments.items():

    print()
    print("-" * 60)
    print(name)
    print("Features:", len(features))

    X = df[features].to_numpy(
        dtype=np.float32
    )

    scores = cross_validate(
        create_model(),
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "f1_macro",
        ],
        n_jobs=1,
    )

    accuracy_scores = scores[
        "test_accuracy"
    ]

    f1_scores = scores[
        "test_f1_macro"
    ]

    print(
        "Accuracy by fold:",
        np.round(accuracy_scores, 4),
    )

    print(
        "Macro F1 by fold:",
        np.round(f1_scores, 4),
    )

    print(
        "Mean accuracy:",
        accuracy_scores.mean(),
    )

    print(
        "Accuracy std:",
        accuracy_scores.std(),
    )

    print(
        "Mean macro F1:",
        f1_scores.mean(),
    )

    print(
        "Macro F1 std:",
        f1_scores.std(),
    )

    results.append(
        {
            "experiment": name,
            "features": len(features),
            "accuracy_mean": accuracy_scores.mean(),
            "accuracy_std": accuracy_scores.std(),
            "macro_f1_mean": f1_scores.mean(),
            "macro_f1_std": f1_scores.std(),
        }
    )


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

print()
print(
    results_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("5-FOLD CROSS-VALIDATION COMPLETE")