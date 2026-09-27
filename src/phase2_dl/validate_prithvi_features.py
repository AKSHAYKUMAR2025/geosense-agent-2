from pathlib import Path

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------
# Load feature table
# ---------------------------------------------------------

df = pd.read_csv(FEATURE_FILE)

print("=" * 60)
print("GeoSense Phase 2 - Prithvi Feature Validation")
print("=" * 60)

print()
print("Rows   :", len(df))
print("Columns:", len(df.columns))


# ---------------------------------------------------------
# Find Prithvi columns
# ---------------------------------------------------------

prithvi_columns = [
    column
    for column in df.columns
    if column.startswith("prithvi_")
]

print()
print("Prithvi features:", len(prithvi_columns))


# ---------------------------------------------------------
# Basic shape checks
# ---------------------------------------------------------

assert len(df) == 408, (
    f"Expected 408 rows, found {len(df)}"
)

assert len(prithvi_columns) == 4096, (
    f"Expected 4096 Prithvi features, "
    f"found {len(prithvi_columns)}"
)


# ---------------------------------------------------------
# Check location IDs
# ---------------------------------------------------------

if "location_i" in df.columns:

    unique_locations = df["location_i"].nunique()

    print("Unique locations:", unique_locations)

    assert unique_locations == 408, (
        "Location IDs are not unique."
    )


# ---------------------------------------------------------
# Check patch IDs
# ---------------------------------------------------------

assert "patch_id" in df.columns

unique_patches = sorted(
    df["patch_id"].unique()
)

print("Prithvi patches:", unique_patches)

assert unique_patches == [0, 1, 2, 3, 4, 5], (
    "Unexpected patch IDs."
)


# ---------------------------------------------------------
# Check missing values
# ---------------------------------------------------------

missing = df[prithvi_columns].isna().sum().sum()

print("Missing Prithvi values:", missing)

assert missing == 0, (
    "Missing values found in Prithvi features."
)


# ---------------------------------------------------------
# Check infinite values
# ---------------------------------------------------------

values = df[prithvi_columns].to_numpy(
    dtype=np.float32
)

infinite_values = np.isinf(values).sum()

print("Infinite Prithvi values:", infinite_values)

assert infinite_values == 0, (
    "Infinite values found in Prithvi features."
)


# ---------------------------------------------------------
# Check numeric range
# ---------------------------------------------------------

print()
print("Prithvi feature minimum:", values.min())
print("Prithvi feature maximum:", values.max())
print("Prithvi feature mean   :", values.mean())
print("Prithvi feature std    :", values.std())


# ---------------------------------------------------------
# Verify same patch → same embedding
# ---------------------------------------------------------

print()
print("Checking patch embedding consistency...")

for patch_id in unique_patches:

    patch_rows = df[
        df["patch_id"] == patch_id
    ]

    patch_values = patch_rows[
        prithvi_columns
    ].to_numpy(dtype=np.float32)

    reference = patch_values[0]

    differences = np.max(
        np.abs(patch_values - reference)
    )

    print(
        f"Patch {patch_id}: "
        f"{len(patch_rows)} candidates, "
        f"max difference = {differences}"
    )

    assert differences == 0, (
        f"Candidates in patch {patch_id} "
        "do not share the same embedding."
    )


# ---------------------------------------------------------
# Check labels
# ---------------------------------------------------------

if "label" in df.columns:

    print()
    print("Labels:")
    print(df["label"].value_counts().sort_index())

    assert set(df["label"].unique()).issubset(
        {0, 1}
    )


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print()
print("=" * 60)
print("ALL PRITHVI FEATURE CHECKS PASSED")
print("=" * 60)