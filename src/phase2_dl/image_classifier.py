import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F

from prithvi_segmentation import PrithviSegmentationModel


# ============================================================
# GeoSense Phase 2
# Exercise 5 - Image Classifier + Change Detection
# ============================================================

# ============================================================
# Paths
# ============================================================

PRITHVI_DIR = Path(
    "data/processed/prithvi"
)

UNET_IMAGE = Path(
    "data/processed/unet/chennai_unet_2024.tif"
)

UNET_MODEL = Path(
    "models/saved/unet_best.pth"
)

PRITHVI_MODEL = Path(
    "models/saved/prithvi_finetuned.pth"
)


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]

# Six spectral bands used by the current project:
#
# B2  = Blue
# B3  = Green
# B4  = Red
# B8  = NIR
# B11 = SWIR1
# B12 = SWIR2

BAND_NAMES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12"
]

# Manual test location
TEST_LAT = 13.0827
TEST_LON = 80.2707
TEST_RADIUS_M = 500

# Change threshold specified by the manual
CHANGE_THRESHOLD = 0.15

# The current imagery is 2024, so we use:
#
# t1 = Q1 2024
# t4 = Q4 2024
#
# as an adapted temporal comparison.

EARLY_IMAGE = (
    PRITHVI_DIR
    / "chennai_prithvi_t1.tif"
)

LATE_IMAGE = (
    PRITHVI_DIR
    / "chennai_prithvi_t4.tif"
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Global model cache
# ============================================================

_MODEL = None
_MODEL_TYPE = None


# ============================================================
# Load Prithvi model
# ============================================================

def _load_prithvi():

    global _MODEL
    global _MODEL_TYPE

    if _MODEL is not None:

        return _MODEL

    if not PRITHVI_MODEL.exists():

        raise FileNotFoundError(
            "Prithvi fine-tuned model not found:\n"
            f"{PRITHVI_MODEL}"
        )

    print(
        "Loading fine-tuned Prithvi model..."
    )

    model = (
        PrithviSegmentationModel()
    )

    checkpoint = torch.load(
        PRITHVI_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    _MODEL = model
    _MODEL_TYPE = "Prithvi"

    print(
        "Prithvi model loaded successfully."
    )

    return _MODEL


# ============================================================
# Load U-Net fallback
# ============================================================

def _load_unet():

    global _MODEL
    global _MODEL_TYPE

    if _MODEL is not None:

        return _MODEL

    if not UNET_MODEL.exists():

        raise FileNotFoundError(
            "U-Net model not found:\n"
            f"{UNET_MODEL}"
        )

    print(
        "Loading U-Net fallback..."
    )

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        UNET_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    _MODEL = model
    _MODEL_TYPE = "U-Net"

    print(
        "U-Net fallback loaded successfully."
    )

    return _MODEL


# ============================================================
# Model loader
# ============================================================

def _load_model():

    # --------------------------------------------------------
    # Requirement:
    #
    # Prefer Prithvi.
    # Fall back to U-Net.
    # --------------------------------------------------------

    if PRITHVI_MODEL.exists():

        try:

            return _load_prithvi()

        except Exception as error:

            print(
                "\nWARNING:"
            )

            print(
                "Prithvi model could not be loaded."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "Falling back to U-Net."
            )

            return _load_unet()

    return _load_unet()


# ============================================================
# Convert latitude/longitude to raster coordinates
# ============================================================

def _latlon_to_pixel(
    dataset,
    lat,
    lon
):

    # --------------------------------------------------------
    # The current GeoSense U-Net raster was created in
    # geographic WGS84 coordinates (EPSG:4326).
    #
    # Therefore latitude/longitude can be used directly.
    #
    # We intentionally do NOT call rasterio.warp.transform()
    # here because the machine has a PostgreSQL/PostGIS PROJ
    # database conflict.
    # --------------------------------------------------------

    raster_crs = dataset.crs

    if raster_crs is None:

        raise RuntimeError(
            "Raster has no CRS."
        )

    # --------------------------------------------------------
    # The GeoSense 2024 raster is expected to be WGS84.
    #
    # Check the CRS without asking PROJ to transform it.
    # --------------------------------------------------------

    crs_string = str(
        raster_crs
    )

    if (
        "4326" not in crs_string
        and
        "WGS 84" not in crs_string
        and
        "WGS84" not in crs_string
    ):

        raise RuntimeError(
            "This classifier expects the current "
            "GeoSense raster to use WGS84/EPSG:4326.\n"
            f"Raster CRS reported as: {raster_crs}"
        )

    # --------------------------------------------------------
    # Direct longitude/latitude lookup.
    #
    # dataset.index() does not require an external CRS
    # transformation when x/y are already in the raster CRS.
    # --------------------------------------------------------

    row, column = dataset.index(
        lon,
        lat
    )

    return (
        row,
        column
    )

    # --------------------------------------------------------
    # Current rasters are in WGS84.
    # --------------------------------------------------------

    if dataset.crs is None:

        raise RuntimeError(
            "Raster has no CRS."
        )

    # --------------------------------------------------------
    # The project rasters are WGS84.
    #
    # If another CRS is encountered, transform coordinates.
    # --------------------------------------------------------

    if dataset.crs.to_epsg() != 4326:

        from rasterio.warp import transform

        x_values, y_values = transform(
            "EPSG:4326",
            dataset.crs,
            [lon],
            [lat]
        )

        x = x_values[0]
        y = y_values[0]

    else:

        x = lon
        y = lat

    row, column = (
        dataset.index(
            x,
            y
        )
    )

    return (
        row,
        column
    )


# ============================================================
# Extract 224x224 image patch
# ============================================================

def _get_image_patch(
    lat,
    lon,
    radius_m=500,
    raster_path=UNET_IMAGE
):

    if not raster_path.exists():

        raise FileNotFoundError(
            "Raster not found:\n"
            f"{raster_path}"
        )

    with rasterio.open(
        raster_path
    ) as dataset:

        # ----------------------------------------------------
        # Check bands
        # ----------------------------------------------------

        if dataset.count < 6:

            raise RuntimeError(
                f"Expected at least 6 bands, "
                f"found {dataset.count}"
            )

        # ----------------------------------------------------
        # Convert location
        # ----------------------------------------------------

        row, column = (
            _latlon_to_pixel(
                dataset,
                lat,
                lon
            )
        )

        # ----------------------------------------------------
        # Determine raster resolution
        # ----------------------------------------------------

        pixel_width = abs(
            dataset.transform.a
        )

        pixel_height = abs(
            dataset.transform.e
        )

        # ----------------------------------------------------
        # Convert 500m radius to pixel radius.
        #
        # This assumes the current project raster is WGS84.
        # For the Chennai study area, this provides a local
        # approximation sufficient for this exercise.
        # ----------------------------------------------------

        lat_scale = 111320.0

        lon_scale = (
            111320.0
            * np.cos(
                np.deg2rad(
                    lat
                )
            )
        )

        metres_per_pixel_x = (
            pixel_width
            * lon_scale
        )

        metres_per_pixel_y = (
            pixel_height
            * lat_scale
        )

        radius_pixels_x = max(
            1,
            int(
                radius_m
                / metres_per_pixel_x
            )
        )

        radius_pixels_y = max(
            1,
            int(
                radius_m
                / metres_per_pixel_y
            )
        )

        # ----------------------------------------------------
        # Create window
        # ----------------------------------------------------

        width = (
            radius_pixels_x
            * 2
        )

        height = (
            radius_pixels_y
            * 2
        )

        window = Window(
            col_off=(
                column
                - radius_pixels_x
            ),
            row_off=(
                row
                - radius_pixels_y
            ),
            width=width,
            height=height
        )

        # ----------------------------------------------------
        # Read six required bands
        # ----------------------------------------------------

        patch = dataset.read(
            indexes=[
                1,
                2,
                3,
                4,
                5,
                6
            ],
            window=window,
            boundless=True,
            fill_value=np.nan
        ).astype(
            np.float32
        )

    # --------------------------------------------------------
    # Fill NaNs with band mean
    # --------------------------------------------------------

    for band_index in range(
        patch.shape[0]
    ):

        band = patch[
            band_index
        ]

        valid = np.isfinite(
            band
        )

        if valid.any():

            mean_value = (
                band[
                    valid
                ].mean()
            )

        else:

            mean_value = 0.0

        band[
            ~valid
        ] = mean_value

    # --------------------------------------------------------
    # Resize to 224x224
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        patch
    ).unsqueeze(
        0
    )

    tensor = F.interpolate(
        tensor,
        size=(
            224,
            224
        ),
        mode="bilinear",
        align_corners=False
    )

    patch = (
        tensor.squeeze(
            0
        )
        .numpy()
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if patch.shape != (
        6,
        224,
        224
    ):

        raise RuntimeError(
            "Unexpected patch shape: "
            f"{patch.shape}"
        )

    return patch


# ============================================================
# Calculate NDVI and NDWI
# ============================================================

def _compute_ndvi_ndwi(
    patch
):

    # --------------------------------------------------------
    # Band indexes
    #
    # patch[0] = B2
    # patch[1] = B3
    # patch[2] = B4
    # patch[3] = B8
    # patch[4] = B11
    # patch[5] = B12
    # --------------------------------------------------------

    green = patch[1]

    red = patch[2]

    nir = patch[3]

    # --------------------------------------------------------
    # NDVI
    #
    # (NIR - Red) / (NIR + Red)
    # --------------------------------------------------------

    ndvi_denominator = (
        nir + red
    )

    ndvi = np.divide(
        nir - red,
        ndvi_denominator,
        out=np.zeros_like(
            nir,
            dtype=np.float32
        ),
        where=(
            np.abs(
                ndvi_denominator
            ) > 1e-8
        )
    )

    # --------------------------------------------------------
    # NDWI
    #
    # (Green - NIR) / (Green + NIR)
    # --------------------------------------------------------

    ndwi_denominator = (
        green + nir
    )

    ndwi = np.divide(
        green - nir,
        ndwi_denominator,
        out=np.zeros_like(
            green,
            dtype=np.float32
        ),
        where=(
            np.abs(
                ndwi_denominator
            ) > 1e-8
        )
    )

    return (
        ndvi,
        ndwi
    )


# ============================================================
# Classify imagery
# ============================================================

def classify_imagery(
    lat,
    lon,
    radius_m=500
):

    start_time = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Load preferred model
    # --------------------------------------------------------

    model = _load_model()

    # --------------------------------------------------------
    # Extract patch
    # --------------------------------------------------------

    patch = _get_image_patch(
        lat,
        lon,
        radius_m
    )

    # --------------------------------------------------------
    # Compute spectral indices
    # --------------------------------------------------------

    ndvi, ndwi = (
        _compute_ndvi_ndwi(
            patch
        )
    )

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    input_tensor = (
        torch.from_numpy(
            patch
        )
        .float()
        .unsqueeze(
            0
        )
        .to(
            DEVICE
        )
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    # --------------------------------------------------------
    # Average probabilities over
    # all pixels.
    # --------------------------------------------------------

    class_probabilities = (
        probabilities
        .mean(
            dim=(
                2,
                3
            )
        )
        .squeeze(
            0
        )
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Normalize to exactly 1
    # --------------------------------------------------------

    class_probabilities = (
        class_probabilities
        / class_probabilities.sum()
    )

    # --------------------------------------------------------
    # Dominant class
    # --------------------------------------------------------

    dominant_index = int(
        np.argmax(
            class_probabilities
        )
    )

    dominant_class = (
        CLASS_NAMES[
            dominant_index
        ]
    )

    confidence = float(
        class_probabilities[
            dominant_index
        ]
    )

    # --------------------------------------------------------
    # Per-class distribution
    # --------------------------------------------------------

    class_distribution = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_distribution[
            class_name
        ] = float(
            class_probabilities[
                index
            ]
        )

    # --------------------------------------------------------
    # Execution time
    # --------------------------------------------------------

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    result = {

        "latitude":
            float(lat),

        "longitude":
            float(lon),

        "radius_m":
            float(radius_m),

        "model":
            _MODEL_TYPE,

        "dominant_class":
            dominant_class,

        "confidence":
            confidence,

        "class_distribution":
            class_distribution,

        "ndvi_mean":
            float(
                np.mean(
                    ndvi
                )
            ),

        "ndvi_min":
            float(
                np.min(
                    ndvi
                )
            ),

        "ndvi_max":
            float(
                np.max(
                    ndvi
                )
            ),

        "ndwi_mean":
            float(
                np.mean(
                    ndwi
                )
            ),

        "ndwi_min":
            float(
                np.min(
                    ndwi
                )
            ),

        "ndwi_max":
            float(
                np.max(
                    ndwi
                )
            ),

        "execution_time_seconds":
            float(
                elapsed_seconds
            )
    }

    return result


# ============================================================
# Print classification result
# ============================================================

def print_classification_result(
    result
):

    print(
        "\n" + "=" * 70
    )

    print(
        "IMAGE CLASSIFICATION RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Latitude: "
        f"{result['latitude']}"
    )

    print(
        f"Longitude: "
        f"{result['longitude']}"
    )

    print(
        f"Radius: "
        f"{result['radius_m']} m"
    )

    print(
        f"Model: "
        f"{result['model']}"
    )

    print(
        f"\nDominant class: "
        f"{result['dominant_class']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        "\nPer-class distribution:"
    )

    for class_name, value in (
        result[
            "class_distribution"
        ].items()
    ):

        print(
            f"  {class_name:15s}: "
            f"{value * 100:.2f}%"
        )

    print(
        "\nSpectral indices:"
    )

    print(
        f"  NDVI mean: "
        f"{result['ndvi_mean']:.4f}"
    )

    print(
        f"  NDVI min:  "
        f"{result['ndvi_min']:.4f}"
    )

    print(
        f"  NDVI max:  "
        f"{result['ndvi_max']:.4f}"
    )

    print(
        f"  NDWI mean: "
        f"{result['ndwi_mean']:.4f}"
    )

    print(
        f"  NDWI min:  "
        f"{result['ndwi_min']:.4f}"
    )

    print(
        f"  NDWI max:  "
        f"{result['ndwi_max']:.4f}"
    )

    print(
        f"\nExecution time: "
        f"{result['execution_time_seconds']:.3f} seconds"
    )

    if (
        result[
            "execution_time_seconds"
        ]
        < 3.0
    ):

        print(
            "Under 3-second target: YES"
        )

    else:

        print(
            "Under 3-second target: NO"
        )


# ============================================================
# Change detection
# ============================================================

def _get_ndvi_patch(
    raster_path,
    lat,
    lon,
    radius_m=500
):

    patch = _get_image_patch(
        lat,
        lon,
        radius_m,
        raster_path
    )

    ndvi, ndwi = (
        _compute_ndvi_ndwi(
            patch
        )
    )

    return (
        ndvi,
        ndwi
    )


def detect_change(
    lat,
    lon,
    radius_m=500
):

    print(
        "\n" + "=" * 70
    )

    print(
        "CHANGE DETECTION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Check both temporal rasters
    # --------------------------------------------------------

    if not EARLY_IMAGE.exists():

        raise FileNotFoundError(
            f"Early image not found:\n"
            f"{EARLY_IMAGE}"
        )

    if not LATE_IMAGE.exists():

        raise FileNotFoundError(
            f"Late image not found:\n"
            f"{LATE_IMAGE}"
        )

    # --------------------------------------------------------
    # Calculate NDVI for Q1 2024
    # --------------------------------------------------------

    early_ndvi, _ = (
        _get_ndvi_patch(
            EARLY_IMAGE,
            lat,
            lon,
            radius_m
        )
    )

    # --------------------------------------------------------
    # Calculate NDVI for Q4 2024
    # --------------------------------------------------------

    late_ndvi, _ = (
        _get_ndvi_patch(
            LATE_IMAGE,
            lat,
            lon,
            radius_m
        )
    )

    # --------------------------------------------------------
    # Pixel-level absolute NDVI difference
    # --------------------------------------------------------

    difference = np.abs(
        late_ndvi
        - early_ndvi
    )

    mean_difference = float(
        np.mean(
            difference
        )
    )

    max_difference = float(
        np.max(
            difference
        )
    )

    changed_pixels = (
        difference
        > CHANGE_THRESHOLD
    )

    changed_percentage = float(
        changed_pixels.mean()
        * 100.0
    )

    # --------------------------------------------------------
    # Flag:
    #
    # True if mean NDVI change exceeds
    # the requested threshold.
    # --------------------------------------------------------

    change_flag = (
        mean_difference
        > CHANGE_THRESHOLD
    )

    print(
        "\nTemporal comparison:"
    )

    print(
        "  Early image: Q1 2024"
    )

    print(
        "  Late image:  Q4 2024"
    )

    print(
        f"\nEarly NDVI mean: "
        f"{early_ndvi.mean():.4f}"
    )

    print(
        f"Late NDVI mean:  "
        f"{late_ndvi.mean():.4f}"
    )

    print(
        f"Mean absolute NDVI difference: "
        f"{mean_difference:.4f}"
    )

    print(
        f"Maximum absolute NDVI difference: "
        f"{max_difference:.4f}"
    )

    print(
        f"Pixels above threshold: "
        f"{changed_percentage:.2f}%"
    )

    print(
        f"Threshold: "
        f"{CHANGE_THRESHOLD}"
    )

    print(
        f"\nChange flag: "
        f"{change_flag}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "This is an adapted Q1-2024 vs Q4-2024 "
        "comparison."
    )

    print(
        "It is NOT the 2015 vs 2023 comparison "
        "specified in the original manual."
    )

    return {

        "early_period":
            "Q1 2024",

        "late_period":
            "Q4 2024",

        "early_ndvi_mean":
            float(
                early_ndvi.mean()
            ),

        "late_ndvi_mean":
            float(
                late_ndvi.mean()
            ),

        "mean_absolute_ndvi_difference":
            mean_difference,

        "max_absolute_ndvi_difference":
            max_difference,

        "changed_pixels_percentage":
            changed_percentage,

        "threshold":
            CHANGE_THRESHOLD,

        "change_flag":
            bool(change_flag),

        "adaptation_note":
            "Q1 2024 vs Q4 2024 proxy; "
            "not 2015 vs 2023."
    }


# ============================================================
# Main test
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GeoSense Phase 2 - Exercise 5"
    )

    print(
        "Image Classifier and Change Detection"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Test classifier
    # --------------------------------------------------------

    print(
        "\nRunning required classifier test..."
    )

    result = classify_imagery(
        TEST_LAT,
        TEST_LON,
        TEST_RADIUS_M
    )

    print_classification_result(
        result
    )

    # --------------------------------------------------------
    # Change detection
    # --------------------------------------------------------

    change_result = detect_change(
        TEST_LAT,
        TEST_LON,
        TEST_RADIUS_M
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EXERCISE 5 TEST COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nClassifier output:"
    )

    print(
        f"Dominant class: "
        f"{result['dominant_class']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Model: "
        f"{result['model']}"
    )

    print(
        f"Execution time: "
        f"{result['execution_time_seconds']:.3f} seconds"
    )

    print(
        "\nChange detection:"
    )

    print(
        f"Change flag: "
        f"{change_result['change_flag']}"
    )

    print(
        f"Mean NDVI difference: "
        f"{change_result['mean_absolute_ndvi_difference']:.4f}"
    )

    print(
        "\nExercise 5 test finished."
    )


if __name__ == "__main__":

    main()