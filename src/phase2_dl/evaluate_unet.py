import numpy as np
import torch
import segmentation_models_pytorch as smp

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# ============================================================
# GeoSense Phase 2
# U-Net Validation Evaluation
# ============================================================

CHIPS_DIR = Path(
    "data/processed/unet/chips"
)

MODEL_FILE = Path(
    "models/saved/unet_best.pth"
)

OUTPUT_DIR = Path(
    "outputs/reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_CLASSES = 5
IN_CHANNELS = 6

BATCH_SIZE = 1

CLASS_NAMES = [
    "Urban",
    "Vegetation",
    "Water",
    "Bare Land",
    "Agriculture"
]


# ============================================================
# Dataset
# ============================================================

class SatelliteDataset(Dataset):

    def __init__(
        self,
        image_files,
        label_files
    ):

        self.image_files = image_files
        self.label_files = label_files

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, index):

        image = np.load(
            self.image_files[index]
        ).astype(
            np.float32
        )

        label = np.load(
            self.label_files[index]
        ).astype(
            np.int64
        )

        # Fill invalid values
        for band in range(
            image.shape[0]
        ):

            valid = np.isfinite(
                image[band]
            )

            if np.any(valid):

                mean_value = np.mean(
                    image[band][valid]
                )

                image[band][~valid] = (
                    mean_value
                )

        image = torch.from_numpy(
            image
        ).float()

        label = torch.from_numpy(
            label
        ).long()

        return image, label


# ============================================================
# Confusion matrix
# ============================================================

def update_confusion_matrix(
    confusion,
    predictions,
    targets
):

    predictions = predictions.flatten()
    targets = targets.flatten()

    for true_class, predicted_class in zip(
        targets,
        predictions
    ):

        true_class = int(
            true_class
        )

        predicted_class = int(
            predicted_class
        )

        if (
            0 <= true_class < NUM_CLASSES
            and
            0 <= predicted_class < NUM_CLASSES
        ):

            confusion[
                true_class,
                predicted_class
            ] += 1


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("GeoSense Phase 2 - U-Net Evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Find chips
    # --------------------------------------------------------

    image_files = sorted(
        CHIPS_DIR.glob(
            "image_*.npy"
        )
    )

    label_files = sorted(
        CHIPS_DIR.glob(
            "label_*.npy"
        )
    )

    if len(image_files) != len(
        label_files
    ):

        raise RuntimeError(
            "Image and label counts differ."
        )

    print(
        f"Total chips: "
        f"{len(image_files)}"
    )

    # --------------------------------------------------------
    # Reproduce the same 80/20 split
    # used during training.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    indices = np.arange(
        len(image_files)
    )

    rng.shuffle(
        indices
    )

    validation_size = int(
        len(indices) * 0.20
    )

    validation_indices = indices[
        -validation_size:
    ]

    validation_images = [
        image_files[i]
        for i in validation_indices
    ]

    validation_labels = [
        label_files[i]
        for i in validation_indices
    ]

    dataset = SatelliteDataset(
        validation_images,
        validation_labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"Validation chips: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Load U-Net
    # --------------------------------------------------------

    print(
        "\nLoading best U-Net model..."
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=device
        )
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    total_correct = 0
    total_pixels = 0

    print(
        "\nRunning validation..."
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_pixels += (
                labels.numel()
            )

            update_confusion_matrix(
                confusion,
                predictions.cpu().numpy(),
                labels.cpu().numpy()
            )

    # --------------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------------

    overall_accuracy = (
        total_correct
        / total_pixels
    )

    print("\n" + "=" * 60)
    print("U-NET EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nOverall pixel accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Print confusion matrix
    # --------------------------------------------------------

    print(
        "\nConfusion Matrix"
    )

    print(
        "Rows = True class"
    )

    print(
        "Columns = Predicted class\n"
    )

    print(
        confusion
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    print(
        "\nPer-class metrics"
    )

    print(
        "-" * 60
    )

    ious = []

    for class_id in range(
        NUM_CLASSES
    ):

        true_positive = (
            confusion[
                class_id,
                class_id
            ]
        )

        false_positive = (
            confusion[
                :,
                class_id
            ].sum()
            - true_positive
        )

        false_negative = (
            confusion[
                class_id,
                :
            ].sum()
            - true_positive
        )

        denominator = (
            true_positive
            + false_positive
            + false_negative
        )

        if denominator > 0:

            iou = (
                true_positive
                / denominator
            )

        else:

            iou = 0.0

        class_total = (
            confusion[
                class_id,
                :
            ].sum()
        )

        if class_total > 0:

            class_accuracy = (
                true_positive
                / class_total
            )

        else:

            class_accuracy = 0.0

        ious.append(
            iou
        )

        print(
            f"{CLASS_NAMES[class_id]:12s} | "
            f"Accuracy: "
            f"{class_accuracy * 100:6.2f}% | "
            f"IoU: "
            f"{iou * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Mean IoU
    # --------------------------------------------------------

    mean_iou = np.mean(
        ious
    )

    print(
        "-" * 60
    )

    print(
        f"Mean IoU: "
        f"{mean_iou * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_file = (
        OUTPUT_DIR
        / "unet_confusion_matrix.csv"
    )

    np.savetxt(
        confusion_file,
        confusion,
        delimiter=",",
        fmt="%d"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_file = (
        OUTPUT_DIR
        / "unet_metrics.txt"
    )

    with open(
        metrics_file,
        "w"
    ) as f:

        f.write(
            "GeoSense Phase 2 U-Net Evaluation\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"Overall Accuracy: "
            f"{overall_accuracy:.6f}\n"
        )

        f.write(
            f"Mean IoU: "
            f"{mean_iou:.6f}\n\n"
        )

        for class_id in range(
            NUM_CLASSES
        ):

            f.write(
                f"{CLASS_NAMES[class_id]} "
                f"IoU: "
                f"{ious[class_id]:.6f}\n"
            )

    # --------------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        confusion
    )

    plt.title(
        "U-Net Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES
    )

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            plt.text(
                j,
                i,
                str(
                    confusion[i, j]
                ),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    confusion_plot = (
        OUTPUT_DIR
        / "unet_confusion_matrix.png"
    )

    plt.savefig(
        confusion_plot,
        dpi=200
    )

    plt.close()

    print(
        f"\nConfusion matrix:"
        f"\n{confusion_file}"
    )

    print(
        f"\nMetrics:"
        f"\n{metrics_file}"
    )

    print(
        f"\nPlot:"
        f"\n{confusion_plot}"
    )

    print(
        "\nU-NET EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()