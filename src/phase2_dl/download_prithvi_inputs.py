from pathlib import Path
import ee
import geemap


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "prithvi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Earth Engine
# ---------------------------------------------------------
EE_PROJECT = "ee-makzaro134242354"

ee.Initialize(project=EE_PROJECT)

print("Earth Engine connection: OK")


# ---------------------------------------------------------
# 3. Chennai study area
# ---------------------------------------------------------
study_area = ee.Geometry.Rectangle([
    80.199,
    13.036,
    80.316,
    13.116
])

print("Chennai study area created.")


# ---------------------------------------------------------
# 4. Sentinel-2 cloud masking
# ---------------------------------------------------------
def mask_sentinel2(image):
    qa = image.select("QA60")

    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit).eq(0))
    )

    return image.updateMask(mask)


# ---------------------------------------------------------
# 5. Sentinel-2 collection
# ---------------------------------------------------------
collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(study_area)
    .filterDate("2024-01-01", "2024-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    .map(mask_sentinel2)
)

print(
    "Sentinel-2 images found:",
    collection.size().getInfo()
)


# ---------------------------------------------------------
# 6. Prithvi bands
# ---------------------------------------------------------
bands = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
]


# ---------------------------------------------------------
# 7. Four temporal periods
# ---------------------------------------------------------
periods = [
    ("2024-01-01", "2024-03-31"),
    ("2024-04-01", "2024-06-30"),
    ("2024-07-01", "2024-09-30"),
    ("2024-10-01", "2024-12-31"),
]


# ---------------------------------------------------------
# 8. Download each image directly
# ---------------------------------------------------------
for index, (start_date, end_date) in enumerate(periods, start=1):

    print()
    print("=" * 50)
    print(f"TIME STEP {index}")
    print(f"{start_date} -> {end_date}")
    print("=" * 50)

    period_collection = (
        collection
        .filterDate(start_date, end_date)
    )

    count = period_collection.size().getInfo()

    print("Images available:", count)

    if count == 0:
        raise RuntimeError(
            f"No Sentinel-2 images found for "
            f"{start_date} to {end_date}"
        )

    composite = (
        period_collection
        .median()
        .select(bands)
        .clip(study_area)
    )

    output_file = (
        OUTPUT_DIR /
        f"chennai_prithvi_t{index}.tif"
    )

    print("Downloading...")
    print("Output:", output_file)

    geemap.ee_export_image(
        composite,
        filename=str(output_file),
        scale=20,
        region=study_area,
        file_per_band=False
    )

    print("Downloaded successfully.")


print()
print("=" * 50)
print("ALL FOUR PRITHVI INPUT IMAGES DOWNLOADED")
print("=" * 50)