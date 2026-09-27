import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import segmentation_models_pytorch as smp

from prithvi_segmentation import PrithviSegmentationModel


# ============================================================
# GeoSense Phase 2
# Exercise 4 - Model Evaluation
#
# U-Net vs Fine-Tuned Prithvi
# ============================================================


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Paths
# ============================================================

CHIP_DIR = Path(
    "data/processed/unet/chips"
)

MODEL_DIR = Path(
    "models/saved"
)

OUTPUT_DIR = Path(
    "outputs/reports"
)

PLOT_DIR = Path(
    "outputs/plots"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
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

IMAGE_SIZE = 224

BATCH_SIZE = 1

VAL_RATIO = 0.20

NUM_WORKERS = 0


# ============================================================
# Dataset
# ============================================================

class SatelliteDataset(Dataset):

    def __init__(
        self,
        image_files,
        label_files
    ):

        if len(image_files) != len(label_files):

            raise ValueError(
                "Image and label counts do not match."
            )

        self.image_files = list(
            image_files
        )

        self.label_files = list(
            label_files
        )

    def __len__(self):

        return len(
            self.image_files
        )

    def __getitem__(
        self,
        index
    ):

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

        if image.shape != (
            6,
            224,
            224
        ):

            raise ValueError(
                f"Unexpected image shape: "
                f"{image.shape}"
            )

        if label.shape != (
            224,
            224
        ):

            raise ValueError(
                f"Unexpected label shape: "
                f"{label.shape}"
            )

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return (
            torch.from_numpy(image),
            torch.from_numpy(label)
        )


# ============================================================
# Find image/label pairs
# ============================================================

def find_chip_pairs():

    image_files = sorted(
        CHIP_DIR.glob(
            "image_*.npy"
        )
    )

    valid_images = []

    valid_labels = []

    for image_path in image_files:

        number = (
            image_path.stem
            .replace(
                "image_",
                ""
            )
        )

        label_path = (
            CHIP_DIR
            / f"label_{number}.npy"
        )

        if label_path.exists():

            valid_images.append(
                image_path
            )

            valid_labels.append(
                label_path
            )

    if len(valid_images) == 0:

        raise FileNotFoundError(
            "No image/label pairs found."
        )

    return (
        valid_images,
        valid_labels
    )


# ============================================================
# Build identical validation split
# ============================================================

def create_validation_loader():

    (
        image_files,
        label_files
    ) = find_chip_pairs()

    dataset = SatelliteDataset(
        image_files,
        label_files
    )

    total = len(
        dataset
    )

    val_size = max(
        1,
        int(
            total * VAL_RATIO
        )
    )

    train_size = (
        total - val_size
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    (
        train_dataset,
        val_dataset
    ) = random_split(
        dataset,
        [
            train_size,
            val_size
        ],
        generator=generator
    )

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        loader,
        len(train_dataset),
        len(val_dataset)
    )


# ============================================================
# Load U-Net
# ============================================================

def load_unet(
    device
):

    print(
        "\nLoading U-Net..."
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=NUM_CLASSES
    )

    checkpoint_path = (
        MODEL_DIR
        / "unet_best.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"U-Net checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support either a raw state_dict or a checkpoint dict.
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

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
        device
    )

    model.eval()

    print(
        "U-Net loaded successfully."
    )

    return model


# ============================================================
# Load Prithvi
# ============================================================

def load_prithvi(
    device
):

    print(
        "\nLoading fine-tuned Prithvi..."
    )

    model = (
        PrithviSegmentationModel()
    )

    checkpoint_path = (
        MODEL_DIR
        / "prithvi_finetuned.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Prithvi checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Our fine-tuning script saved a dictionary containing:
    #
    # model_state_dict
    # optimizer_state_dict
    # etc.
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

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
        device
    )

    model.eval()

    print(
        "Fine-tuned Prithvi loaded successfully."
    )

    return model


# ============================================================
# Confusion matrix
# ============================================================

def update_confusion_matrix(
    confusion,
    predictions,
    labels
):

    predictions = (
        predictions
        .reshape(-1)
        .cpu()
        .numpy()
    )

    labels = (
        labels
        .reshape(-1)
        .cpu()
        .numpy()
    )

    for true_class, predicted_class in zip(
        labels,
        predictions
    ):

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
# Calculate metrics
# ============================================================

def calculate_metrics(
    confusion
):

    total_pixels = (
        confusion.sum()
    )

    correct_pixels = (
        np.trace(
            confusion
        )
    )

    if total_pixels > 0:

        pixel_accuracy = (
            correct_pixels
            / total_pixels
        )

    else:

        pixel_accuracy = 0.0

    ious = []

    class_accuracies = []

    for class_index in range(
        NUM_CLASSES
    ):

        true_positive = (
            confusion[
                class_index,
                class_index
            ]
        )

        false_positive = (
            confusion[
                :,
                class_index
            ].sum()
            - true_positive
        )

        false_negative = (
            confusion[
                class_index,
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
                class_index,
                :
            ].sum()
        )

        if class_total > 0:

            accuracy = (
                true_positive
                / class_total
            )

        else:

            accuracy = 0.0

        ious.append(
            iou
        )

        class_accuracies.append(
            accuracy
        )

    mean_iou = float(
        np.mean(
            ious
        )
    )

    return {
        "pixel_accuracy":
            float(pixel_accuracy),

        "mean_iou":
            mean_iou,

        "ious":
            ious,

        "class_accuracies":
            class_accuracies
    }


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    loader,
    device,
    model_name
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"Evaluating: {model_name}"
    )

    print(
        "=" * 70
    )

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    total_batches = len(
        loader
    )

    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(
            loader,
            start=1
        ):

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

            update_confusion_matrix(
                confusion,
                predictions,
                labels
            )

            print(
                f"Processed validation chip "
                f"{batch_index}/{total_batches}"
            )

    metrics = calculate_metrics(
        confusion
    )

    print(
        "\nResults"
    )

    print(
        "-" * 70
    )

    print(
        f"Pixel Accuracy: "
        f"{metrics['pixel_accuracy'] * 100:.2f}%"
    )

    print(
        f"Mean IoU: "
        f"{metrics['mean_iou'] * 100:.2f}%"
    )

    print(
        "\nPer-class results:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:15s} "
            f"IoU = "
            f"{metrics['ious'][index] * 100:.2f}% "
            f"| Accuracy = "
            f"{metrics['class_accuracies'][index] * 100:.2f}%"
        )

    print(
        "\nConfusion matrix:"
    )

    print(
        confusion
    )

    return (
        metrics,
        confusion
    )


# ============================================================
# Save confusion matrix CSV
# ============================================================

def save_confusion_csv(
    confusion,
    model_name
):

    safe_name = (
        model_name.lower()
        .replace(
            " ",
            "_"
        )
        .replace(
            "-",
            "_"
        )
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_name}_confusion_matrix.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Actual \\ Predicted"
            ]
            + CLASS_NAMES
        )

        for index, class_name in enumerate(
            CLASS_NAMES
        ):

            writer.writerow(
                [
                    class_name
                ]
                + confusion[
                    index
                ].tolist()
            )

    return output_path


# ============================================================
# Save metrics CSV
# ============================================================

def save_metrics_csv(
    all_results
):

    output_path = (
        OUTPUT_DIR
        / "phase2_model_comparison.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "Model",
                "Pixel Accuracy",
                "Mean IoU",
                "Urban IoU",
                "Vegetation IoU",
                "Water IoU",
                "Bare Land IoU",
                "Agriculture IoU"
            ]
        )

        for model_name, result in (
            all_results.items()
        ):

            metrics = result[
                "metrics"
            ]

            writer.writerow(
                [
                    model_name,
                    metrics[
                        "pixel_accuracy"
                    ],
                    metrics[
                        "mean_iou"
                    ],
                    metrics[
                        "ious"
                    ][0],
                    metrics[
                        "ious"
                    ][1],
                    metrics[
                        "ious"
                    ][2],
                    metrics[
                        "ious"
                    ][3],
                    metrics[
                        "ious"
                    ][4]
                ]
            )

    return output_path


# ============================================================
# Save text report
# ============================================================

def save_text_report(
    all_results,
    train_size,
    val_size
):

    output_path = (
        OUTPUT_DIR
        / "phase2_model_comparison.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "GeoSense Phase 2 Model Evaluation\n"
        )

        file.write(
            "===================================\n\n"
        )

        file.write(
            f"Training chips: {train_size}\n"
        )

        file.write(
            f"Validation chips: {val_size}\n"
        )

        file.write(
            "Validation split seed: 42\n\n"
        )

        for model_name, result in (
            all_results.items()
        ):

            metrics = result[
                "metrics"
            ]

            confusion = result[
                "confusion"
            ]

            file.write(
                f"{model_name}\n"
            )

            file.write(
                "-" * 50
                + "\n"
            )

            file.write(
                f"Pixel Accuracy: "
                f"{metrics['pixel_accuracy'] * 100:.4f}%\n"
            )

            file.write(
                f"Mean IoU: "
                f"{metrics['mean_iou'] * 100:.4f}%\n\n"
            )

            for index, class_name in enumerate(
                CLASS_NAMES
            ):

                file.write(
                    f"{class_name}: "
                    f"IoU = "
                    f"{metrics['ious'][index] * 100:.4f}%, "
                    f"Accuracy = "
                    f"{metrics['class_accuracies'][index] * 100:.4f}%\n"
                )

            file.write(
                "\nConfusion Matrix:\n"
            )

            for row in confusion:

                file.write(
                    " ".join(
                        str(value)
                        for value in row
                    )
                    + "\n"
                )

            file.write(
                "\n\n"
            )

    return output_path


# ============================================================
# Plot confusion matrix
# ============================================================

def save_confusion_plot(
    confusion,
    model_name
):

    import matplotlib.pyplot as plt

    safe_name = (
        model_name.lower()
        .replace(
            " ",
            "_"
        )
        .replace(
            "-",
            "_"
        )
    )

    output_path = (
        PLOT_DIR
        / f"{safe_name}_confusion_matrix.png"
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            7
        )
    )

    image = ax.imshow(
        confusion
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "Actual Class"
    )

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    # Add cell values
    for row in range(
        NUM_CLASSES
    ):

        for column in range(
            NUM_CLASSES
        ):

            ax.text(
                column,
                row,
                str(
                    confusion[
                        row,
                        column
                    ]
                ),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "GeoSense Phase 2 - "
        "U-Net vs Prithvi Evaluation"
    )

    print(
        "=" * 70
    )

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
    # Validation loader
    # --------------------------------------------------------

    print(
        "\nPreparing validation dataset..."
    )

    (
        validation_loader,
        train_size,
        val_size
    ) = create_validation_loader()

    print(
        f"Training chips: "
        f"{train_size}"
    )

    print(
        f"Validation chips: "
        f"{val_size}"
    )

    # --------------------------------------------------------
    # Evaluate U-Net
    # --------------------------------------------------------

    unet = load_unet(
        device
    )

    (
        unet_metrics,
        unet_confusion
    ) = evaluate_model(
        unet,
        validation_loader,
        device,
        "U-Net"
    )

    # Free memory before loading Prithvi
    del unet

    if device.type == "cuda":

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Evaluate Prithvi
    # --------------------------------------------------------

    prithvi = load_prithvi(
        device
    )

    (
        prithvi_metrics,
        prithvi_confusion
    ) = evaluate_model(
        prithvi,
        validation_loader,
        device,
        "Prithvi"
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    all_results = {

        "U-Net": {
            "metrics":
                unet_metrics,
            "confusion":
                unet_confusion
        },

        "Prithvi": {
            "metrics":
                prithvi_metrics,
            "confusion":
                prithvi_confusion
        }
    }

    # --------------------------------------------------------
    # Save confusion matrices
    # --------------------------------------------------------

    unet_csv = save_confusion_csv(
        unet_confusion,
        "U-Net"
    )

    prithvi_csv = save_confusion_csv(
        prithvi_confusion,
        "Prithvi"
    )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    unet_plot = save_confusion_plot(
        unet_confusion,
        "U-Net"
    )

    prithvi_plot = save_confusion_plot(
        prithvi_confusion,
        "Prithvi"
    )

    # --------------------------------------------------------
    # Save comparison CSV
    # --------------------------------------------------------

    comparison_csv = (
        save_metrics_csv(
            all_results
        )
    )

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    report_path = (
        save_text_report(
            all_results,
            train_size,
            val_size
        )
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "phase2_model_comparison.json"
    )

    json_results = {}

    for model_name, result in (
        all_results.items()
    ):

        json_results[
            model_name
        ] = {

            "pixel_accuracy":
                result["metrics"][
                    "pixel_accuracy"
                ],

            "mean_iou":
                result["metrics"][
                    "mean_iou"
                ],

            "ious": {
                CLASS_NAMES[index]:
                    value
                for index, value
                in enumerate(
                    result["metrics"][
                        "ious"
                    ]
                )
            },

            "class_accuracy": {
                CLASS_NAMES[index]:
                    value
                for index, value
                in enumerate(
                    result["metrics"][
                        "class_accuracies"
                    ]
                )
            }
        }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_results,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    for model_name, result in (
        all_results.items()
    ):

        metrics = result[
            "metrics"
        ]

        print(
            f"\n{model_name}"
        )

        print(
            f"Pixel Accuracy: "
            f"{metrics['pixel_accuracy'] * 100:.2f}%"
        )

        print(
            f"Mean IoU: "
            f"{metrics['mean_iou'] * 100:.2f}%"
        )

        for index, class_name in enumerate(
            CLASS_NAMES
        ):

            print(
                f"  {class_name}: "
                f"{metrics['ious'][index] * 100:.2f}% IoU"
            )

    print(
        "\nSaved evaluation files:"
    )

    print(
        comparison_csv
    )

    print(
        report_path
    )

    print(
        json_path
    )

    print(
        unet_csv
    )

    print(
        prithvi_csv
    )

    print(
        unet_plot
    )

    print(
        prithvi_plot
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()