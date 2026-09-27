import ee
import geemap
import rasterio
from pathlib import Path


# ============================================================
# GeoSense Phase 2
# Exercise 3 - U-Net Data Preparation
# ============================================================

PROJECT_ID = "ee-makzaro134242354"

BBOX = [
    80.199,
    13.036,
    80.316,
    13.116
]

OUTPUT_DIR = Path("data/processed/unet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
CLOUD_PERCENT = 30


def initialize_gee():
    """Initialize Google Earth Engine."""

    ee.Initialize(project=PROJECT_ID)

    print("Earth Engine: OK")


def create_study_area():
    """Create Chennai study area."""

    xmin, ymin, xmax, ymax = BBOX

    return ee.Geometry.Rectangle(
        [xmin, ymin, xmax, ymax]
    )


def mask_sentinel2(image):
    """
    Mask Sentinel-2 clouds using QA60.
    """

    qa = image.select("QA60")

    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit).eq(0)
        .And(
            qa.bitwiseAnd(cirrus_bit).eq(0)
        )
    )

    return (
        image
        .updateMask(mask)
        .divide(10000)
        .copyProperties(
            image,
            ["system:time_start"]
        )
    )


def get_sentinel_collection(region):
    """Retrieve Sentinel-2 imagery."""

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(region)
        .filterDate(
            START_DATE,
            END_DATE
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                CLOUD_PERCENT
            )
        )
        .map(mask_sentinel2)
    )

    return collection


def create_composite(region):
    """
    Create a median Sentinel-2 composite.

    Six channels are retained:
        B2  Blue
        B3  Green
        B4  Red
        B8  NIR
        B11 SWIR1
        B12 SWIR2
    """

    collection = get_sentinel_collection(region)

    count = collection.size().getInfo()

    print(
        f"Sentinel-2 images found: {count}"
    )

    if count == 0:
        raise RuntimeError(
            "No Sentinel-2 images were found."
        )

    composite = (
        collection
        .median()
        .select(
            [
                "B2",
                "B3",
                "B4",
                "B8",
                "B11",
                "B12"
            ]
        )
        .clip(region)
    )

    return composite


def add_indices(image):
    """
    Add NDVI and NDWI bands.

    NDVI = (NIR - Red) / (NIR + Red)
    NDWI = (Green - NIR) / (Green + NIR)
    """

    ndvi = image.normalizedDifference(
        ["B8", "B4"]
    ).rename("NDVI")

    ndwi = image.normalizedDifference(
        ["B3", "B8"]
    ).rename("NDWI")

    return image.addBands(
        [ndvi, ndwi]
    )


def export_image(image, region, output_path):
    """Export the composite from Earth Engine."""

    print(
        f"Exporting:\n{output_path}"
    )

    geemap.ee_export_image(
        image,
        filename=str(output_path),
        scale=10,
        region=region,
        file_per_band=False
    )

    print("Export completed.")


def inspect_raster(path):
    """Check the downloaded GeoTIFF."""

    with rasterio.open(path) as src:

        print("\nRaster verification")
        print("-" * 45)

        print("Width :", src.width)
        print("Height:", src.height)
        print("Bands :", src.count)
        print("CRS   :", src.crs)
        print("Dtype :", src.dtypes[0])

        print(
            "Band descriptions:",
            src.descriptions
        )


def main():

    print("=" * 60)
    print("GeoSense Phase 2")
    print("U-Net Satellite Data Preparation")
    print("=" * 60)

    initialize_gee()

    region = create_study_area()

    print(
        "Chennai study area created."
    )

    composite = create_composite(
        region
    )

    composite = add_indices(
        composite
    )

    output_path = (
        OUTPUT_DIR
        / "chennai_unet_2024.tif"
    )

    export_image(
        composite,
        region,
        output_path
    )

    inspect_raster(
        output_path
    )

    print("\nTASK 3.1B: COMPLETED")
    print(
        "Six-band Sentinel-2 imagery "
        "with NDVI and NDWI was prepared."
    )


if __name__ == "__main__":
    main()