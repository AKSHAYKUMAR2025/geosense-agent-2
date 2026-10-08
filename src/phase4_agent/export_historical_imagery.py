import os
from pathlib import Path

import ee
import requests


# ============================================================
# GeoSense Phase 4
# E.4.3 Historical Sentinel-2 Imagery
# ============================================================


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_ID = "ee-makzaro134242354"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_imagery"
)


# ============================================================
# 2. SICILY / VULCANO STUDY AREA
# ============================================================
#
# These coordinates are the actual WGS84 bounds obtained
# from the Phase 3 LiDAR dataset.
#
# Longitude:
#   14.957943 -> 14.961042
#
# Latitude:
#   38.413989 -> 38.416862
#

WEST = 14.957943
SOUTH = 38.413989
EAST = 14.961042
NORTH = 38.416862


# ============================================================
# 3. HISTORICAL YEARS
# ============================================================

YEARS = [
    2015,
    2018,
    2021,
    2024,
]

CLOUD_PERCENT = 30


# ============================================================
# 4. INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

print("========================================")
print("GeoSense Historical Imagery Export")
print("========================================")

print("\nInitializing Google Earth Engine...")

try:

    ee.Initialize(
        project=PROJECT_ID
    )

    print(
        "Earth Engine connection: OK"
    )

except Exception as e:

    print(
        "Earth Engine initialization failed."
    )

    print(e)

    raise


# ============================================================
# 5. CREATE STUDY AREA
# ============================================================

STUDY_AREA = ee.Geometry.Rectangle(
    [
        WEST,
        SOUTH,
        EAST,
        NORTH,
    ]
)

print("\nStudy area:")
print(
    f"West={WEST}, "
    f"South={SOUTH}, "
    f"East={EAST}, "
    f"North={NORTH}"
)


# ============================================================
# 6. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# 7. SENTINEL-2 CLOUD MASK
# ============================================================

def mask_sentinel2(image):

    qa60 = image.select(
        "QA60"
    )

    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    mask = (
        qa60
        .bitwiseAnd(cloud_bit)
        .eq(0)
        .And(
            qa60
            .bitwiseAnd(cirrus_bit)
            .eq(0)
        )
    )

    return image.updateMask(
        mask
    )


# ============================================================
# 8. PROCESS EACH YEAR
# ============================================================

for year in YEARS:

    print("\n========================================")
    print(
        f"Processing Sentinel-2 year: {year}"
    )
    print("========================================")


    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    start_date = (
        f"{year}-01-01"
    )

    end_date = (
        f"{year + 1}-01-01"
    )

    print(
        f"Date range: "
        f"{start_date} -> {end_date}"
    )


    # --------------------------------------------------------
    # Load Sentinel-2
    # --------------------------------------------------------

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_HARMONIZED"
        )
        .filterBounds(
            STUDY_AREA
        )
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                CLOUD_PERCENT
            )
        )
        .map(
            mask_sentinel2
        )
    )


    # --------------------------------------------------------
    # Count available images
    # --------------------------------------------------------

    image_count = (
        collection
        .size()
        .getInfo()
    )

    print(
        f"Sentinel-2 images found: "
        f"{image_count}"
    )


    # --------------------------------------------------------
    # No imagery
    # --------------------------------------------------------

    if image_count == 0:

        print(
            f"WARNING: No Sentinel-2 imagery "
            f"was found for {year}."
        )

        continue


    # --------------------------------------------------------
    # Create median composite
    # --------------------------------------------------------

    print(
        "Creating median composite..."
    )

    composite = (
        collection
        .median()
        .clip(
            STUDY_AREA
        )
    )


    print(
        "Median composite created."
    )


    # --------------------------------------------------------
    # Select natural-colour bands
    #
    # B4 = Red
    # B3 = Green
    # B2 = Blue
    #
    # Sentinel-2 reflectance is scaled by 10000.
    # --------------------------------------------------------

    rgb = (
        composite
        .select(
            [
                "B4",
                "B3",
                "B2",
            ]
        )
        .divide(10000)
    )


    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"sicily_{year}_rgb.tif"
    )


    print(
        "\nPreparing download:"
    )

    print(
        output_file
    )


    # --------------------------------------------------------
    # Get Earth Engine download URL
    # --------------------------------------------------------

    region = (
        STUDY_AREA
        .getInfo()
        ["coordinates"]
    )


    download_url = (
        rgb.getDownloadURL(
            {
                "name": (
                    f"sicily_{year}_rgb"
                ),

                "region": region,

                "scale": 10,

                "crs": "EPSG:4326",

                "filePerBand": False,

                "format": "GEO_TIFF",
            }
        )
    )


    print(
        "Download URL generated."
    )


    # --------------------------------------------------------
    # Download GeoTIFF
    # --------------------------------------------------------

    response = requests.get(
        download_url,
        timeout=300
    )


    print(
        "HTTP status:",
        response.status_code
    )


    response.raise_for_status()


    with open(
        output_file,
        "wb"
    ) as file:

        file.write(
            response.content
        )


    print(
        "Saved:"
    )

    print(
        output_file
    )


    # --------------------------------------------------------
    # Verify file
    # --------------------------------------------------------

    if output_file.exists():

        file_size = (
            output_file.stat()
            .st_size
        )

        print(
            f"File size: "
            f"{file_size:,} bytes"
        )

    else:

        raise RuntimeError(
            f"Expected output file was not created: "
            f"{output_file}"
        )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("HISTORICAL IMAGERY EXPORT COMPLETE")
print("========================================")

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nRequested years:"
)

print(
    YEARS
)

print(
    "\nNext E.4.3 step:"
)

print(
    "Convert the GeoTIFF files to Cesium "
    "XYZ tiles."
)