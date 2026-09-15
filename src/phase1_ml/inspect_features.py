import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)


print("=" * 60)
print("FEATURE DATA INSPECTION")
print("=" * 60)

df = pd.read_csv(FEATURE_FILE)

print()
print(f"Total rows: {len(df)}")

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Data types:")
print(df.dtypes)

print()
print("Missing values:")
print(df.isnull().sum())

print()
print("Feature statistics:")
print(df.describe())

print()
print("First 10 rows:")
print(df.head(10))

print()
print("=" * 60)
print("FEATURE INSPECTION COMPLETE")
print("=" * 60)