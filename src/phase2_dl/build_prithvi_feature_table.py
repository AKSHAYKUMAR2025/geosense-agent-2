from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prithvi_patch_embeddings.npy"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_prithvi_patch_mapping.csv"
)

PHASE1_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labelled_sites.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prithvi_features.csv"
)


# ---------------------------------------------------------
# Load Prithvi embeddings
# ---------------------------------------------------------

embeddings = np.load(EMBEDDINGS_FILE)

print("Prithvi embeddings shape:", embeddings.shape)

if embeddings.shape != (6, 4096):
    raise ValueError(
        f"Unexpected embedding shape: {embeddings.shape}. "
        "Expected (6, 4096)."
    )


# ---------------------------------------------------------
# Load candidate → patch mapping
# ---------------------------------------------------------

mapping = pd.read_csv(MAPPING_FILE)

print("Mapped candidates:", len(mapping))


# ---------------------------------------------------------
# Load Phase 1 labels/features
# ---------------------------------------------------------

phase1 = pd.read_csv(PHASE1_FILE)

print("Phase 1 rows:", len(phase1))

if len(phase1) != 408:
    raise ValueError(
        f"Expected 408 Phase 1 rows, found {len(phase1)}."
    )


# ---------------------------------------------------------
# Verify location IDs
# ---------------------------------------------------------

mapping["location_i"] = mapping["location_i"].astype(int)

if "location_id" in phase1.columns:
    phase1_id_column = "location_id"
elif "location_i" in phase1.columns:
    phase1_id_column = "location_i"
else:
    raise ValueError(
        "Could not find location ID column in labelled_sites.csv."
    )

phase1[phase1_id_column] = phase1[phase1_id_column].astype(int)


# ---------------------------------------------------------
# Merge mapping with Phase 1 data
# ---------------------------------------------------------

merged = mapping.merge(
    phase1,
    left_on="location_i",
    right_on=phase1_id_column,
    how="inner",
    suffixes=("", "_phase1"),
)


print("Merged rows:", len(merged))

if len(merged) != 408:
    raise ValueError(
        f"Expected 408 merged rows, found {len(merged)}."
    )


# ---------------------------------------------------------
# Add Prithvi features
# ---------------------------------------------------------

for feature_index in range(4096):

    merged[f"prithvi_{feature_index:04d}"] = (
        merged["patch_id"]
        .map(
            lambda patch_id:
            embeddings[int(patch_id), feature_index]
        )
    )


# ---------------------------------------------------------
# Basic validation
# ---------------------------------------------------------

prithvi_columns = [
    f"prithvi_{i:04d}"
    for i in range(4096)
]

missing_values = (
    merged[prithvi_columns]
    .isna()
    .sum()
    .sum()
)

print("Missing Prithvi values:", missing_values)

if missing_values != 0:
    raise ValueError(
        "Missing values detected in Prithvi features."
    )


# ---------------------------------------------------------
# Save feature table
# ---------------------------------------------------------

merged.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print()
print("=" * 60)
print("PRITHVI FEATURE TABLE CREATED")
print("=" * 60)

print("Rows       :", len(merged))
print("Prithvi features:", len(prithvi_columns))
print("Total columns:", len(merged.columns))

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("PRITHVI FEATURE TABLE COMPLETE")