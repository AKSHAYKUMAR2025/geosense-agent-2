import numpy as np
import rasterio
from pathlib import Path


# ============================================================
# GeoSense Phase 2
# Exercise 3 - U-Net Chip Generation
# ============================================================

INPUT_IMAGE = Path(
    "data/processed/unet/chennai_unet_2024.tif"
)

INPUT_LABELS = Path(
    "data/processed/unet/chennai_labels_2024.tif"
)

OUTPUT_DIR = Path(
    "data/processed/unet/chips"
)

CHIP_SIZE = 224
STRIDE = 174
MAX_NAN_RATIO = 0.20


def fill_nan_with_band_mean(chip):
    """
    Replace remaining NaN values in each band
    with that band's mean value.
    """

    chip = chip.copy()

    for band in range(chip.shape[0]):

        band_data = chip[band]

        valid = np.isfinite(band_data)

        if not np.any(valid):
            continue

        mean_value = np.nanmean(
            band_data
        )

        band_data[~valid] = mean_value

        chip[band] = band_data

    return chip


def main():

    print("=" * 60)
    print("GeoSense Phase 2")
    print("U-Net 224x224 Chip Generation")
    print("=" * 60)

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{INPUT_IMAGE}"
        )

    if not INPUT_LABELS.exists():
        raise FileNotFoundError(
            f"Label raster not found:\n{INPUT_LABELS}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    print("\nReading satellite image...")

    with rasterio.open(INPUT_IMAGE) as src:

        image = src.read()

        profile = src.profile.copy()

        height = src.height
        width = src.width

    print(
        f"Image shape: {image.shape}"
    )

    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    print("\nReading label raster...")

    with rasterio.open(INPUT_LABELS) as src:

        labels = src.read(1)

    print(
        f"Label shape: {labels.shape}"
    )

    if image.shape[1:] != labels.shape:
        raise ValueError(
            "Image and label dimensions do not match."
        )

    # --------------------------------------------------------
    # We use only the six spectral bands.
    #
    # Bands:
    # 1 B2
    # 2 B3
    # 3 B4
    # 4 B8
    # 5 B11
    # 6 B12
    #
    # Bands 7 and 8 are NDVI and NDWI and are not
    # included as U-Net input channels.
    # --------------------------------------------------------

    image = image[:6]

    print(
        f"\nU-Net input channels: "
        f"{image.shape[0]}"
    )

    print(
        "Expected: 6"
    )

    # --------------------------------------------------------
    # Calculate number of possible positions
    # --------------------------------------------------------

    y_positions = range(
        0,
        height - CHIP_SIZE + 1,
        STRIDE
    )

    x_positions = range(
        0,
        width - CHIP_SIZE + 1,
        STRIDE
    )

    total_possible = (
        len(list(y_positions))
        * len(list(x_positions))
    )

    print(
        f"\nPossible chips: "
        f"{total_possible}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    saved = 0
    skipped_nan = 0
    skipped_empty = 0

    class_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }

    # --------------------------------------------------------
    # Generate chips
    # --------------------------------------------------------

    print("\nGenerating chips...")

    chip_id = 0

    for y in range(
        0,
        height - CHIP_SIZE + 1,
        STRIDE
    ):

        for x in range(
            0,
            width - CHIP_SIZE + 1,
            STRIDE
        ):

            image_chip = image[
                :,
                y:y + CHIP_SIZE,
                x:x + CHIP_SIZE
            ]

            label_chip = labels[
                y:y + CHIP_SIZE,
                x:x + CHIP_SIZE
            ]

            # ----------------------------------------------
            # Calculate NaN ratio
            # ----------------------------------------------

            nan_ratio = (
                np.isnan(image_chip).any(
                    axis=0
                ).mean()
            )

            if nan_ratio > MAX_NAN_RATIO:

                skipped_nan += 1
                continue

            # ----------------------------------------------
            # Fill remaining NaNs
            # ----------------------------------------------

            image_chip = (
                fill_nan_with_band_mean(
                    image_chip
                )
            )

            # ----------------------------------------------
            # Check valid classes
            # ----------------------------------------------

            valid_labels = label_chip[
                label_chip != 255
            ]

            if valid_labels.size == 0:

                skipped_empty += 1
                continue

            # ----------------------------------------------
            # Save input chip
            # ----------------------------------------------

            image_file = (
                OUTPUT_DIR
                / f"image_{chip_id:04d}.npy"
            )

            label_file = (
                OUTPUT_DIR
                / f"label_{chip_id:04d}.npy"
            )

            np.save(
                image_file,
                image_chip.astype(
                    np.float32
                )
            )

            np.save(
                label_file,
                label_chip.astype(
                    np.uint8
                )
            )

            # ----------------------------------------------
            # Count classes
            # ----------------------------------------------

            for class_id in class_counts:

                class_counts[class_id] += int(
                    np.sum(
                        label_chip == class_id
                    )
                )

            saved += 1
            chip_id += 1

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CHIP GENERATION SUMMARY")
    print("=" * 60)

    print(
        f"Possible chips : {total_possible}"
    )

    print(
        f"Saved chips    : {saved}"
    )

    print(
        f"Skipped NaN    : {skipped_nan}"
    )

    print(
        f"Skipped empty  : {skipped_empty}"
    )

    print("\nClass pixels across saved chips:")

    for class_id, count in class_counts.items():

        print(
            f"Class {class_id}: "
            f"{count:,}"
        )

    print("\nOutput directory:")

    print(
        OUTPUT_DIR
    )

    if saved == 0:

        raise RuntimeError(
            "No chips were generated."
        )

    print("\n" + "=" * 60)
    print("TASK 3.1C: CHIP GENERATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()