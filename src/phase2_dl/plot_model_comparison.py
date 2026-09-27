import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# GeoSense Phase 2
# Training Curve Visualization
# ============================================================

EVALUATION_DIR = Path(
    "models/evaluation"
)

OUTPUT_DIR = Path(
    "outputs/plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


UNET_HISTORY = (
    EVALUATION_DIR
    / "unet_history.json"
)

PRITHVI_HISTORY = (
    EVALUATION_DIR
    / "prithvi_history.json"
)


# ============================================================
# Load history
# ============================================================

def load_history(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            f"History file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "GeoSense Phase 2 - "
        "Training Curve Visualization"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load histories
    # --------------------------------------------------------

    unet = load_history(
        UNET_HISTORY
    )

    prithvi = load_history(
        PRITHVI_HISTORY
    )

    print(
        f"\nU-Net epochs: "
        f"{len(unet['train_loss'])}"
    )

    print(
        f"Prithvi epochs: "
        f"{len(prithvi['train_loss'])}"
    )

    # ========================================================
    # Prithvi Loss Curve
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        range(
            1,
            len(
                prithvi["train_loss"]
            ) + 1
        ),
        prithvi["train_loss"],
        marker="o",
        label="Prithvi Training Loss"
    )

    plt.plot(
        range(
            1,
            len(
                prithvi["val_loss"]
            ) + 1
        ),
        prithvi["val_loss"],
        marker="o",
        label="Prithvi Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Prithvi Fine-Tuning Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    prithvi_loss_path = (
        OUTPUT_DIR
        / "prithvi_loss_curve.png"
    )

    plt.savefig(
        prithvi_loss_path,
        dpi=200
    )

    plt.close()

    print(
        f"\nSaved:\n"
        f"{prithvi_loss_path}"
    )

    # ========================================================
    # Prithvi Accuracy Curve
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        range(
            1,
            len(
                prithvi["train_acc"]
            ) + 1
        ),
        [
            value * 100
            for value in prithvi["train_acc"]
        ],
        marker="o",
        label="Prithvi Training Accuracy"
    )

    plt.plot(
        range(
            1,
            len(
                prithvi["val_acc"]
            ) + 1
        ),
        [
            value * 100
            for value in prithvi["val_acc"]
        ],
        marker="o",
        label="Prithvi Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Pixel Accuracy (%)"
    )

    plt.title(
        "Prithvi Fine-Tuning Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    prithvi_accuracy_path = (
        OUTPUT_DIR
        / "prithvi_accuracy_curve.png"
    )

    plt.savefig(
        prithvi_accuracy_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved:\n"
        f"{prithvi_accuracy_path}"
    )

    # ========================================================
    # U-Net Loss Curve
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        range(
            1,
            len(
                unet["train_loss"]
            ) + 1
        ),
        unet["train_loss"],
        label="U-Net Training Loss"
    )

    plt.plot(
        range(
            1,
            len(
                unet["val_loss"]
            ) + 1
        ),
        unet["val_loss"],
        label="U-Net Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "U-Net Training Loss"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    unet_loss_path = (
        OUTPUT_DIR
        / "unet_loss_curve.png"
    )

    plt.savefig(
        unet_loss_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved:\n"
        f"{unet_loss_path}"
    )

    # ========================================================
    # U-Net Accuracy Curve
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        range(
            1,
            len(
                unet["train_acc"]
            ) + 1
        ),
        [
            value * 100
            for value in unet["train_acc"]
        ],
        label="U-Net Training Accuracy"
    )

    plt.plot(
        range(
            1,
            len(
                unet["val_acc"]
            ) + 1
        ),
        [
            value * 100
            for value in unet["val_acc"]
        ],
        label="U-Net Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Pixel Accuracy (%)"
    )

    plt.title(
        "U-Net Training Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    unet_accuracy_path = (
        OUTPUT_DIR
        / "unet_accuracy_curve.png"
    )

    plt.savefig(
        unet_accuracy_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved:\n"
        f"{unet_accuracy_path}"
    )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING CURVES CREATED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()