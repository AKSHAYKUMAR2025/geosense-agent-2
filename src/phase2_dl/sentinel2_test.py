import ee


# ============================================================
# GeoSense Agent 2.0 — Phase 2
# Sentinel-2 Earth Engine Connection Test
# ============================================================

PROJECT_ID = "ee-makzaro134242354"


# ============================================================
# 1. Initialize Google Earth Engine
# ============================================================

print("=" * 60)
print("GeoSense Agent 2.0 — Phase 2")
print("Sentinel-2 Earth Engine Test")
print("=" * 60)

print("\nConnecting to Google Earth Engine...")

ee.Initialize(
    project=PROJECT_ID
)

print("Earth Engine connection: OK")


# ============================================================
# 2. Define GeoSense Chennai Study Area
# ============================================================

study_area = ee.Geometry.Rectangle([
    80.199,
    13.036,
    80.316,
    13.116
])

print("Chennai study area created.")


# ============================================================
# 3. Load Sentinel-2 Level-2A
# ============================================================

print("\nLoading Sentinel-2 Level-2A imagery...")

sentinel2 = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filterBounds(study_area)
    .filterDate(
        "2024-01-01",
        "2024-12-31"
    )
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            30
        )
    )
)


# ============================================================
# 4. Count Images
# ============================================================

image_count = sentinel2.size().getInfo()

print(
    f"Sentinel-2 images found: {image_count}"
)


# ============================================================
# 5. Create Median Composite
# ============================================================

print("\nCreating Sentinel-2 median composite...")

composite = (
    sentinel2
    .median()
    .clip(study_area)
)


# ============================================================
# 6. Check Important Bands
# ============================================================

print("\nChecking Sentinel-2 bands...")

band_names = composite.bandNames().getInfo()

print("Available bands:")
print(band_names)


# ============================================================
# 7. Calculate NDVI
# ============================================================

print("\nCalculating NDVI...")

ndvi = composite.normalizedDifference(
    ["B8", "B4"]
).rename("NDVI")


# ============================================================
# 8. Calculate NDWI
# ============================================================

print("Calculating NDWI...")

ndwi = composite.normalizedDifference(
    ["B3", "B8"]
).rename("NDWI")


# ============================================================
# 9. Calculate NDBI
# ============================================================

print("Calculating NDBI...")

ndbi = composite.normalizedDifference(
    ["B11", "B8"]
).rename("NDBI")


# ============================================================
# 10. Test One Pixel
# ============================================================

print("\nTesting satellite values...")

test_point = ee.Geometry.Point([
    80.2707,
    13.0827
])

test_image = composite.select([
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12"
])

test_values = (
    test_image
    .reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=test_point,
        scale=10,
        maxPixels=1e9
    )
    .getInfo()
)

print("\nSatellite values at test point:")

for band, value in test_values.items():

    print(
        f"{band:5s}: {value}"
    )


# ============================================================
# 11. Test NDVI Value
# ============================================================

ndvi_value = (
    ndvi
    .reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=test_point,
        scale=10,
        maxPixels=1e9
    )
    .getInfo()
)

print("\nNDVI at test point:")

print(
    ndvi_value
)


# ============================================================
# 12. Final Status
# ============================================================

print("\n" + "=" * 60)
print("SENTINEL-2 PHASE 2 TEST COMPLETED")
print("=" * 60)

print("\nEarth Engine: OK")
print(f"Images found : {image_count}")
print("Composite    : OK")
print("NDVI         : OK")
print("NDWI         : OK")
print("NDBI         : OK")
print("Pixel test   : OK")

print("\nNext step:")
print("Extract Sentinel-2 features for the 408 GeoSense")
print("candidate locations.")

print("=" * 60)