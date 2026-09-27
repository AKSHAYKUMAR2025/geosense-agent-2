from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx


# ============================================================
# GeoSense Agent 2.0 — Phase 1
# Training Label Visualization
# Esri Satellite Basemap
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Input and Output Files
# ============================================================

LABELLED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labelled_sites.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "training_labels_chennai.png"
)


# ============================================================
# Main Function
# ============================================================

def main():

    print("=" * 60)
    print("GeoSense Agent 2.0 — Phase 1")
    print("Training Label Map")
    print("Esri Satellite Basemap")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Check labelled dataset
    # --------------------------------------------------------

    print("\nChecking labelled dataset...")

    if not LABELLED_FILE.exists():

        print("ERROR: labelled_sites.csv was not found.")
        print(f"Expected location: {LABELLED_FILE}")

        return

    print("Input file found:")
    print(LABELLED_FILE)

    # --------------------------------------------------------
    # 2. Load labelled dataset
    # --------------------------------------------------------

    print("\nLoading labelled candidate locations...")

    df = pd.read_csv(LABELLED_FILE)

    print(f"Loaded {len(df)} labelled locations.")

    # --------------------------------------------------------
    # 3. Check required columns
    # --------------------------------------------------------

    required_columns = [
        "latitude",
        "longitude",
        "label"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Required columns are missing:")
        print(missing_columns)

        return

    print("Required columns found:")
    print(required_columns)

    # --------------------------------------------------------
    # 4. Create geographic points
    # --------------------------------------------------------

    print("\nCreating geographic points...")

    candidates = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"]
        ),
        crs="EPSG:4326"
    )

    # --------------------------------------------------------
    # 5. Separate Good and Poor sites
    # --------------------------------------------------------

    good = candidates[
        candidates["label"] == 1
    ].copy()

    poor = candidates[
        candidates["label"] == 0
    ].copy()

    print("\nLabel summary:")
    print(f"Good sites (1): {len(good)}")
    print(f"Poor sites (0): {len(poor)}")
    print(f"Total sites:    {len(candidates)}")

    # --------------------------------------------------------
    # 6. Convert coordinates to Web Mercator
    # --------------------------------------------------------
    # Contextily satellite tiles use EPSG:3857.

    print("\nConverting coordinates for map display...")

    candidates_web = candidates.to_crs(
        epsg=3857
    )

    good_web = good.to_crs(
        epsg=3857
    )

    poor_web = poor.to_crs(
        epsg=3857
    )

    # --------------------------------------------------------
    # 7. Create map
    # --------------------------------------------------------

    print("\nCreating training-label map...")

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    # --------------------------------------------------------
    # 8. Set map extent
    # --------------------------------------------------------

    minx, miny, maxx, maxy = (
        candidates_web.total_bounds
    )

    x_padding = (maxx - minx) * 0.05
    y_padding = (maxy - miny) * 0.05

    ax.set_xlim(
        minx - x_padding,
        maxx + x_padding
    )

    ax.set_ylim(
        miny - y_padding,
        maxy + y_padding
    )

    # --------------------------------------------------------
    # 9. Add Esri World Imagery satellite basemap
    # --------------------------------------------------------

    print("\nAdding Esri satellite basemap...")

    try:

        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldImagery,
            zoom="auto"
        )

        print(
            "Esri satellite basemap added successfully."
        )

    except Exception as error:

        print(
            "\nWARNING: Could not download "
            "Esri satellite basemap."
        )

        print(f"Reason: {error}")

        print(
            "The candidate points will still "
            "be plotted."
        )

    # --------------------------------------------------------
    # 10. Plot Poor Sites
    # --------------------------------------------------------

    print("\nPlotting poor sites...")

    if not poor_web.empty:

        poor_web.plot(
            ax=ax,
            color="red",
            markersize=40,
            alpha=0.90,
            edgecolor="white",
            linewidth=0.6,
            label=f"Poor sites (0): {len(poor_web)}"
        )

    # --------------------------------------------------------
    # 11. Plot Good Sites
    # --------------------------------------------------------

    print("Plotting good sites...")

    if not good_web.empty:

        good_web.plot(
            ax=ax,
            color="lime",
            markersize=45,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.6,
            label=f"Good sites (1): {len(good_web)}"
        )

    # --------------------------------------------------------
    # 12. Add title
    # --------------------------------------------------------

    ax.set_title(
        "GeoSense Agent 2.0 — Phase 1\n"
        "Training Labels — Chennai Study Area",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    # --------------------------------------------------------
    # 13. Add legend
    # --------------------------------------------------------

    ax.legend(
        loc="upper right",
        fontsize=11,
        frameon=True
    )

    # --------------------------------------------------------
    # 14. Remove axes
    # --------------------------------------------------------

    ax.set_axis_off()

    # --------------------------------------------------------
    # 15. Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 16. Save map
    # --------------------------------------------------------

    print("\nSaving map...")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # 17. Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING LABEL MAP CREATED SUCCESSFULLY")
    print("=" * 60)

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nResults:")
    print(f"Total locations : {len(candidates)}")
    print(f"Good sites (1)  : {len(good_web)}")
    print(f"Poor sites (0)  : {len(poor_web)}")

    print("\nBasemap:")
    print("Esri World Imagery")

    print("\nMap generation completed.")
    print("=" * 60)


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()