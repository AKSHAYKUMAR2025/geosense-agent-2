from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILES_TO_CHECK = {
    "Features": PROJECT_ROOT / "data" / "processed" / "features.csv",
    "Labelled sites": PROJECT_ROOT / "data" / "processed" / "labelled_sites.csv",
    "SHAP report": PROJECT_ROOT / "outputs" / "reports" / "shap_importance.png",
    "Saved model": PROJECT_ROOT / "models" / "saved" / "site_scorer_model.pkl",
    "Requirements": PROJECT_ROOT / "requirements.txt",
}


def main():
    print("GeoSense Phase 1 Health Check")
    print("=" * 35)

    all_ok = True

    for name, path in FILES_TO_CHECK.items():
        if path.exists():
            print(f"[OK] {name}: {path}")
        else:
            print(f"[MISSING] {name}: {path}")
            all_ok = False

    print("=" * 35)

    if all_ok:
        print("Phase 1 health check passed.")
    else:
        print("Phase 1 health check found missing files.")

    return all_ok


if __name__ == "__main__":
    main()