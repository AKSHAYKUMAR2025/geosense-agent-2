from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "phase2_results_summary.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


results = pd.DataFrame(
    [
        {
            "experiment": "Phase 1 baseline",
            "features": 3,
            "accuracy": 0.9634146341463414,
            "macro_f1": 0.9563,
            "correct_test_predictions": 79,
            "test_samples": 82,
        },
        {
            "experiment": "Prithvi only",
            "features": 4096,
            "accuracy": 0.5609756097560976,
            "macro_f1": 0.5607,
            "correct_test_predictions": 46,
            "test_samples": 82,
        },
        {
            "experiment": "Phase 1 + Prithvi",
            "features": 4099,
            "accuracy": 0.975609756097561,
            "macro_f1": 0.9724,
            "correct_test_predictions": 80,
            "test_samples": 82,
        },
    ]
)


results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("=" * 60)
print("GeoSense Phase 2 - Results Summary")
print("=" * 60)

print()
print(results.to_string(index=False))

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("PHASE 2 RESULTS SUMMARY COMPLETE")