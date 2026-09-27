import numpy as np
from pathlib import Path
import rasterio


# ============================================================
# GeoSense Phase 2
# Exercise 3 - Task 3.1
# Real Sentinel-2 Pseudo Labelling
# ============================================================

# Class IDs
URBAN = 0
VEGETATION = 1
WATER = 2
BARE_LAND = 3
AGRICULTURE = 4

CLASS_NAMES = {
    URBAN: "Urban",
    VEGETATION: "Vegetation",
    WATER: "Water",
    BARE_LAND: "Bare Land",
    AGRICULTURE: "Agriculture",
}


INPUT_FILE = Path(
    "data/processed/unet/chennai_unet_2024.tif"
)

OUTPUT_DIR = Path(
    "data/processed/unet"
)

LABEL_FILE = (
    OUTPUT_DIR / "chennai_labels_2024.tif"
)


def classify_pixel(ndvi, ndwi, nir, swir):
    """
    Apply the Phase 2 laboratory manual rules.

    NDWI > 0.3
        -> Water

    NDVI > 0.4
        -> Vegetation

    NDVI > 0.15
        -> Agriculture

    NDVI < 0.05 and SWIR > 0.2
        -> Urban

    Otherwise
        -> Bare Land
    """

    if ndwi > 0.3:
        return WATER

    elif ndvi > 0.4:
        return VEGETATION

    elif ndvi > 0.15:
        return AGRICULTURE

    elif ndvi < 0.05 and swir > 0.2:
        return URBAN

    else:
        return BARE_LAND


def create_label_mask(ndvi, ndwi, nir, swir):
    """
    Create a five-class label mask.

    All arrays must have the same dimensions.
    """

    if not (
        ndvi.shape
        == ndwi.shape
        == nir.shape
        == swir.shape
    ):
        raise ValueError(
            "NDVI, NDWI, NIR and SWIR arrays "
            "must have the same shape."
        )

    labels = np.full(
        ndvi.shape,
        BARE_LAND,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 1. Water
    # --------------------------------------------------------

    water = ndwi > 0.3
    labels[water] = WATER

    # --------------------------------------------------------
    # 2. Vegetation
    # --------------------------------------------------------

    vegetation = (
        (~water)
        & (ndvi > 0.4)
    )

    labels[vegetation] = VEGETATION

    # --------------------------------------------------------
    # 3. Agriculture
    # --------------------------------------------------------

    agriculture = (
        (~water)
        & (~vegetation)
        & (ndvi > 0.15)
    )

    labels[agriculture] = AGRICULTURE

    # --------------------------------------------------------
    # 4. Urban
    # --------------------------------------------------------

    urban = (
        (~water)
        & (~vegetation)
        & (~agriculture)
        & (ndvi < 0.05)
        & (swir > 0.2)
    )

    labels[urban] = URBAN

    # Everything remaining stays Bare Land.

    return labels


def print_distribution(labels):
    """
    Print the number and percentage of pixels
    belonging to each class.
    """

    total = labels.size

    print("\nClass Distribution")
    print("=" * 60)

    for class_id, class_name in CLASS_NAMES.items():

        count = int(
            np.sum(labels == class_id)
        )

        percentage = (
            count / total * 100
        )

        print(
            f"{class_id} - "
            f"{class_name:12s}: "
            f"{count:10,d} pixels "
            f"({percentage:6.2f}%)"
        )

    print("-" * 60)
    print(
        f"Total pixels: {total:,}"
    )


def main():

    print("=" * 60)
    print("GeoSense Phase 2")
    print("Real Sentinel-2 Pseudo Labelling")
    print("=" * 60)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Read Sentinel-2 raster
    # --------------------------------------------------------

    print(
        f"\nReading:\n{INPUT_FILE}"
    )

    with rasterio.open(INPUT_FILE) as src:

        print(
            f"Raster size: "
            f"{src.width} x {src.height}"
        )

        print(
            f"Number of bands: "
            f"{src.count}"
        )

        if src.count != 8:

            raise ValueError(
                "Expected 8 bands: "
                "B2, B3, B4, B8, B11, B12, "
                "NDVI, NDWI."
            )

        # ----------------------------------------------------
        # Band order created by prepare_unet_data.py
        #
        # 1 = B2
        # 2 = B3
        # 3 = B4
        # 4 = B8
        # 5 = B11
        # 6 = B12
        # 7 = NDVI
        # 8 = NDWI
        # ----------------------------------------------------

        b3_green = src.read(2)
        b8_nir = src.read(4)
        b11_swir = src.read(5)

        ndvi = src.read(7)
        ndwi = src.read(8)

        profile = src.profile.copy()

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    print("\nChecking raster values...")

    print(
        f"NDVI range: "
        f"{np.nanmin(ndvi):.4f} "
        f"to "
        f"{np.nanmax(ndvi):.4f}"
    )

    print(
        f"NDWI range: "
        f"{np.nanmin(ndwi):.4f} "
        f"to "
        f"{np.nanmax(ndwi):.4f}"
    )

    print(
        f"NIR range: "
        f"{np.nanmin(b8_nir):.4f} "
        f"to "
        f"{np.nanmax(b8_nir):.4f}"
    )

    print(
        f"SWIR range: "
        f"{np.nanmin(b11_swir):.4f} "
        f"to "
        f"{np.nanmax(b11_swir):.4f}"
    )

    # --------------------------------------------------------
    # Create label mask
    # --------------------------------------------------------

    print(
        "\nGenerating five-class label mask..."
    )

    labels = create_label_mask(
        ndvi=ndvi,
        ndwi=ndwi,
        nir=b8_nir,
        swir=b11_swir,
    )

    # --------------------------------------------------------
    # Handle invalid pixels
    # --------------------------------------------------------

    invalid = (
        ~np.isfinite(ndvi)
        | ~np.isfinite(ndwi)
        | ~np.isfinite(b8_nir)
        | ~np.isfinite(b11_swir)
    )

    invalid_count = int(
        np.sum(invalid)
    )

    if invalid_count > 0:

        print(
            f"\nInvalid pixels found: "
            f"{invalid_count:,}"
        )

        # Mark invalid pixels as 255.
        # 255 is reserved as ignore/no-data.
        labels[invalid] = 255

    else:

        print(
            "\nInvalid pixels found: 0"
        )

    # --------------------------------------------------------
    # Print distribution
    # --------------------------------------------------------

    valid_labels = labels[
        labels != 255
    ]

    print_distribution(
        valid_labels
    )

    # --------------------------------------------------------
    # Save label raster
    # --------------------------------------------------------

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=255,
        compress="lzw"
    )

    print(
        f"\nSaving label raster:\n{LABEL_FILE}"
    )

    with rasterio.open(
        LABEL_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(
            labels,
            1
        )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    with rasterio.open(
        LABEL_FILE
    ) as check:

        saved = check.read(1)

    unique_values = np.unique(
        saved
    )

    print(
        "\nSaved label values:"
    )

    print(
        unique_values
    )

    allowed_values = {
        0, 1, 2, 3, 4, 255
    }

    if not set(
        unique_values
    ).issubset(allowed_values):

        raise ValueError(
            "Unexpected class ID found."
        )

    print("\n" + "=" * 60)
    print("TASK 3.1B: REAL LABEL GENERATION PASSED")
    print("=" * 60)

    print(
        f"Output:\n{LABEL_FILE}"
    )


if __name__ == "__main__":
    main()