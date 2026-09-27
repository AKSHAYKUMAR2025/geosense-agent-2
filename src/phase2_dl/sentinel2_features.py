import os
import ee
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_ID = "ee-makzaro134242354"

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "labelled_sites.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sentinel2_features.csv"
)

START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

CLOUD_PERCENT = 30


# ============================================================
# 2. INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

try:
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine connection: OK")
except Exception as e:
    print("Earth Engine initialization failed.")
    print(e)
    raise


# ============================================================
# 3. LOAD PHASE 1 CANDIDATE LOCATIONS
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "location_id",
    "latitude",
    "longitude",
    "label",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print(f"Candidate locations loaded: {len(df)}")


# ============================================================
# 4. CREATE EARTH ENGINE POINTS
# ============================================================

features = []

for _, row in df.iterrows():

    point = ee.Geometry.Point([
        float(row["longitude"]),
        float(row["latitude"])
    ])

    feature = ee.Feature(
        point,
        {
            "location_id": int(row["location_id"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "label": int(row["label"]),
        }
    )

    features.append(feature)


candidate_points = ee.FeatureCollection(features)

print("Earth Engine candidate points created.")


# ============================================================
# 5. SENTINEL-2 CLOUD MASK
# ============================================================

def mask_sentinel2(image):

    qa60 = image.select("QA60")

    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    mask = (
        qa60.bitwiseAnd(cloud_bit).eq(0)
        .And(
            qa60.bitwiseAnd(cirrus_bit).eq(0)
        )
    )

    return image.updateMask(mask)


# ============================================================
# 6. LOAD SENTINEL-2 IMAGERY
# ============================================================

sentinel_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(candidate_points)
    .filterDate(START_DATE, END_DATE)
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            CLOUD_PERCENT
        )
    )
    .map(mask_sentinel2)
)

image_count = sentinel_collection.size().getInfo()

print(f"Sentinel-2 images found: {image_count}")

if image_count == 0:
    raise RuntimeError(
        "No Sentinel-2 images were found."
    )


# ============================================================
# 7. CREATE MEDIAN COMPOSITE
# ============================================================

composite = sentinel_collection.median()

print("Sentinel-2 median composite created.")


# ============================================================
# 8. SELECT BANDS AND CONVERT REFLECTANCE
# ============================================================

# Sentinel-2 surface reflectance is scaled by 10000.
# Divide by 10000 so values are approximately 0-1.

bands = (
    composite
    .select([
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12",
    ])
    .divide(10000)
)


# ============================================================
# 9. CALCULATE SPECTRAL INDICES
# ============================================================

ndvi = (
    bands.select("B8")
    .subtract(bands.select("B4"))
    .divide(
        bands.select("B8")
        .add(bands.select("B4"))
    )
    .rename("NDVI")
)

ndwi = (
    bands.select("B3")
    .subtract(bands.select("B8"))
    .divide(
        bands.select("B3")
        .add(bands.select("B8"))
    )
    .rename("NDWI")
)

ndbi = (
    bands.select("B11")
    .subtract(bands.select("B8"))
    .divide(
        bands.select("B11")
        .add(bands.select("B8"))
    )
    .rename("NDBI")
)


# ============================================================
# 10. CREATE FINAL FEATURE IMAGE
# ============================================================

feature_image = bands.addBands([
    ndvi,
    ndwi,
    ndbi
])


# ============================================================
# 11. SAMPLE ALL 408 LOCATIONS
# ============================================================

# B11 and B12 are 20 m Sentinel-2 bands.
# We therefore use 20 m as the common extraction scale.

samples = feature_image.sampleRegions(
    collection=candidate_points,
    properties=[
        "location_id",
        "latitude",
        "longitude",
        "label",
    ],
    scale=20,
    geometries=False
)


# ============================================================
# 12. DOWNLOAD RESULTS FROM EARTH ENGINE
# ============================================================

print("Extracting Sentinel-2 features...")
print("Please wait...")

sample_data = samples.getInfo()["features"]

print(
    f"Features returned by Earth Engine: "
    f"{len(sample_data)}"
)


# ============================================================
# 13. CONVERT EARTH ENGINE RESULTS TO DATAFRAME
# ============================================================

rows = []

for feature in sample_data:

    properties = feature["properties"]

    rows.append(properties)


sentinel_df = pd.DataFrame(rows)


# ============================================================
# 14. CHECK EXPECTED COLUMNS
# ============================================================

expected_columns = [
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

for column in expected_columns:

    if column not in sentinel_df.columns:
        sentinel_df[column] = pd.NA


# Keep columns in a predictable order.

sentinel_df = sentinel_df[expected_columns]


# ============================================================
# 15. SORT BY LOCATION ID
# ============================================================

sentinel_df = sentinel_df.sort_values(
    "location_id"
).reset_index(drop=True)


# ============================================================
# 16. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(sentinel_df.isna().sum())


# ============================================================
# 17. SAVE OUTPUT
# ============================================================

output_directory = os.path.dirname(OUTPUT_FILE)

if output_directory:
    os.makedirs(
        output_directory,
        exist_ok=True
    )

sentinel_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("SENTINEL-2 FEATURE EXTRACTION COMPLETE")
print("========================================")

print(f"Rows saved : {len(sentinel_df)}")
print(f"Columns    : {len(sentinel_df.columns)}")
print(f"Output     : {OUTPUT_FILE}")

print("\nColumns:")
print(list(sentinel_df.columns))

print("\nFirst 5 rows:")
print(sentinel_df.head())

print("\nEarth Engine: OK")
print("Sentinel-2 : OK")
print("Indices     : NDVI, NDWI, NDBI")
print("Scale       : 20 meters")