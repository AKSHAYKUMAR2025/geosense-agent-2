import pandas as pd
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input file
FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

# Output file
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "labelled_sites.csv"
)


print("=" * 60)
print("CREATING TRAINING LABELS")
print("=" * 60)

# Load feature data
df = pd.read_csv(FEATURE_FILE)

print()
print(f"Loaded {len(df)} candidate locations.")


# ---------------------------------------------------------
# Calculate thresholds from the actual dataset
# ---------------------------------------------------------

road_threshold = df["dist_road_m"].median()
hospital_threshold = df["dist_hospital_m"].median()


print()
print("Label thresholds:")
print(f"Road distance threshold: {road_threshold:.2f} metres")
print(f"Hospital distance threshold: {hospital_threshold:.2f} metres")


# ---------------------------------------------------------
# Create labels
#
# Good site:
# - low flood risk
# - near road
# - near hospital
#
# Bad site:
# - high flood risk
# - OR far from road
# - OR far from hospital
# ---------------------------------------------------------

good_site = (
    (df["flood_risk"] <= 2)
    & (df["dist_road_m"] <= road_threshold)
    & (df["dist_hospital_m"] <= hospital_threshold)
)

df["label"] = good_site.astype(int)


# ---------------------------------------------------------
# Show label counts
# ---------------------------------------------------------

good_count = (df["label"] == 1).sum()
bad_count = (df["label"] == 0).sum()

print()
print("Label results:")
print(f"Good sites (1): {good_count}")
print(f"Bad sites  (0): {bad_count}")


# ---------------------------------------------------------
# Save labelled data
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Labelled data saved to:")
print(OUTPUT_FILE)

print()
print("=" * 60)
print("LABEL CREATION COMPLETE")
print("=" * 60)