import os
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

PHASE1_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "labelled_sites.csv"
)

SENTINEL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sentinel2_features.csv"
)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("Loading Phase 1 dataset...")
phase1 = pd.read_csv(PHASE1_FILE)

print("Loading Sentinel-2 dataset...")
sentinel = pd.read_csv(SENTINEL_FILE)


# ============================================================
# 3. BASIC ROW CHECK
# ============================================================

print("\n========================================")
print("1. ROW COUNT CHECK")
print("========================================")

print(f"Phase 1 rows     : {len(phase1)}")
print(f"Sentinel-2 rows  : {len(sentinel)}")

if len(phase1) == len(sentinel) == 408:
    print("PASS: Both datasets contain 408 rows.")
else:
    print("FAIL: Row counts do not match expected 408.")


# ============================================================
# 4. REQUIRED COLUMN CHECK
# ============================================================

print("\n========================================")
print("2. COLUMN CHECK")
print("========================================")

required_columns = [
    "location_id",
    "latitude",
    "longitude",
    "label",
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "NDWI",
    "NDBI",
]

missing_columns = [
    column
    for column in required_columns
    if column not in sentinel.columns
]

if not missing_columns:
    print("PASS: All required Sentinel-2 columns exist.")
else:
    print("FAIL: Missing columns:")
    print(missing_columns)


# ============================================================
# 5. DUPLICATE LOCATION CHECK
# ============================================================

print("\n========================================")
print("3. DUPLICATE LOCATION CHECK")
print("========================================")

duplicate_count = sentinel["location_id"].duplicated().sum()

print(f"Duplicate location IDs: {duplicate_count}")

if duplicate_count == 0:
    print("PASS: No duplicate location IDs.")
else:
    print("FAIL: Duplicate location IDs found.")


# ============================================================
# 6. LOCATION MATCH CHECK
# ============================================================

print("\n========================================")
print("4. LOCATION MATCH CHECK")
print("========================================")

phase1_locations = phase1[
    [
        "location_id",
        "latitude",
        "longitude"
    ]
].sort_values("location_id").reset_index(drop=True)

sentinel_locations = sentinel[
    [
        "location_id",
        "latitude",
        "longitude"
    ]
].sort_values("location_id").reset_index(drop=True)

location_match = phase1_locations.equals(
    sentinel_locations
)

if location_match:
    print(
        "PASS: location_id, latitude, and longitude "
        "match Phase 1."
    )
else:
    print(
        "FAIL: Location information does not "
        "match Phase 1."
    )


# ============================================================
# 7. LABEL MATCH CHECK
# ============================================================

print("\n========================================")
print("5. LABEL MATCH CHECK")
print("========================================")

phase1_labels = phase1[
    ["location_id", "label"]
].sort_values("location_id").reset_index(drop=True)

sentinel_labels = sentinel[
    ["location_id", "label"]
].sort_values("location_id").reset_index(drop=True)

label_match = phase1_labels.equals(
    sentinel_labels
)

if label_match:
    print("PASS: Phase 1 labels are preserved.")
else:
    print("FAIL: Labels do not match Phase 1.")


# ============================================================
# 8. MISSING VALUE CHECK
# ============================================================

print("\n========================================")
print("6. MISSING VALUE CHECK")
print("========================================")

missing_values = sentinel[
    required_columns
].isna().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("PASS: No missing values.")
else:
    print("FAIL: Missing values found.")


# ============================================================
# 9. REFLECTANCE RANGE CHECK
# ============================================================

print("\n========================================")
print("7. SENTINEL-2 REFLECTANCE CHECK")
print("========================================")

reflectance_columns = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]

print(
    sentinel[reflectance_columns].describe()
)

reflectance_min = sentinel[
    reflectance_columns
].min().min()

reflectance_max = sentinel[
    reflectance_columns
].max().max()

print(f"\nMinimum reflectance: {reflectance_min}")
print(f"Maximum reflectance: {reflectance_max}")

if reflectance_min >= 0 and reflectance_max <= 1:
    print(
        "PASS: Reflectance values are within "
        "the expected 0-1 scale."
    )
else:
    print(
        "WARNING: Some reflectance values fall "
        "outside the expected 0-1 range."
    )


# ============================================================
# 10. SPECTRAL INDEX RANGE CHECK
# ============================================================

print("\n========================================")
print("8. SPECTRAL INDEX CHECK")
print("========================================")

index_columns = [
    "NDVI",
    "NDWI",
    "NDBI",
]

print(
    sentinel[index_columns].describe()
)

for column in index_columns:

    minimum = sentinel[column].min()
    maximum = sentinel[column].max()

    print(
        f"{column}: "
        f"min={minimum:.4f}, "
        f"max={maximum:.4f}"
    )

    if minimum >= -1 and maximum <= 1:
        print(f"  PASS: {column} is within -1 to +1.")
    else:
        print(
            f"  WARNING: {column} has values "
            f"outside -1 to +1."
        )


# ============================================================
# 11. LABEL DISTRIBUTION
# ============================================================

print("\n========================================")
print("9. LABEL DISTRIBUTION")
print("========================================")

print(
    sentinel["label"].value_counts().sort_index()
)

print("\nLabel meanings from Phase 1:")
print("0 = Poor site")
print("1 = Good site")


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

all_basic_checks_passed = (
    len(phase1) == 408
    and len(sentinel) == 408
    and not missing_columns
    and duplicate_count == 0
    and location_match
    and label_match
    and missing_values.sum() == 0
    and reflectance_min >= 0
    and reflectance_max <= 1
)

print("\n========================================")
print("SENTINEL-2 VALIDATION SUMMARY")
print("========================================")

if all_basic_checks_passed:
    print("ALL BASIC CHECKS PASSED")
    print("Sentinel-2 dataset is ready for the next Phase 2 step.")
else:
    print("SOME CHECKS REQUIRE ATTENTION")

print("\nValidation complete.")