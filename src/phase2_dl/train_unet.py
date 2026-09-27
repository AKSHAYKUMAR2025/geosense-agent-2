import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp


# ============================================================
# GeoSense Phase 2
# Exercise 3 - U-Net Training
# ============================================================

CHIPS_DIR = Path(
    "data/processed/unet/chips"
)

MODEL_DIR = Path(
    "models/saved"
)

EVALUATION_DIR = Path(
    "models/evaluation"
)

PLOT_DIR = Path(
    "outputs/plots"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

SEED = 42

NUM_CLASSES = 5
IN_CHANNELS = 6

BATCH_SIZE = 2
EPOCHS = 20

LEARNING_RATE = 0.001

STEP_SIZE = 10
GAMMA = 0.5

VALIDATION_SPLIT = 0.20

NUM_WORKERS = 0


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Dataset
# ============================================================

class SatelliteDataset(Dataset):

    def __init__(
        self,
        image_files,
        label_files,
        augment=False
    ):

        self.image_files = image_files
        self.label_files = label_files
        self.augment = augment

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

        # ----------------------------------------------------
        # Replace invalid image values
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Data augmentation
        # ----------------------------------------------------

        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:

                image = np.flip(
                    image,
                    axis=2
                ).copy()

                label = np.flip(
                    label,
                    axis=1
                ).copy()

            # Vertical flip
            if random.random() < 0.5:

                image = np.flip(
                    image,
                    axis=1
                ).copy()

                label = np.flip(
                    label,
                    axis=0
                ).copy()

            # 90-degree rotation
            if random.random() < 0.5:

                k = random.randint(
                    1,
                    3
                )

                image = np.rot90(
                    image,
                    k=k,
                    axes=(1, 2)
                ).copy()

                label = np.rot90(
                    label,
                    k=k
                ).copy()

            # Brightness augmentation
            if random.random() < 0.5:

                factor = random.uniform(
                    0.9,
                    1.1
                )

                image = image * factor

            # Contrast augmentation
            if random.random() < 0.5:

                factor = random.uniform(
                    0.9,
                    1.1
                )

                mean = image.mean(
                    axis=(1, 2),
                    keepdims=True
                )

                image = (
                    (image - mean)
                    * factor
                    + mean
                )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image
        ).float()

        label_tensor = torch.from_numpy(
            label
        ).long()

        return (
            image_tensor,
            label_tensor
        )


# ============================================================
# Dataset discovery
# ============================================================

def find_pairs():

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

    if len(image_files) == 0:

        raise RuntimeError(
            "No image chips found."
        )

    if len(image_files) != len(
        label_files
    ):

        raise RuntimeError(
            "Image and label counts differ."
        )

    return (
        image_files,
        label_files
    )


# ============================================================
# Accuracy calculation
# ============================================================

def calculate_accuracy(
    predictions,
    targets
):

    correct = (
        predictions == targets
    ).sum()

    total = targets.numel()

    return (
        correct.item()
        / total
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        running_correct += (
            predictions == labels
        ).sum().item()

        running_total += (
            labels.numel()
        )

    average_loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = (
        running_correct
        / running_total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

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

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            running_correct += (
                predictions == labels
            ).sum().item()

            running_total += (
                labels.numel()
            )

    average_loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = (
        running_correct
        / running_total
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# Plot training curves
# ============================================================

def save_training_curves(
    history
):

    epochs = range(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    # Loss
    plt.figure()

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "GeoSense U-Net Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        / "unet_loss_curve.png",
        dpi=200
    )

    plt.close()

    # Accuracy
    plt.figure()

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "GeoSense U-Net Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR
        / "unet_accuracy_curve.png",
        dpi=200
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print(
        "GeoSense Phase 2 - U-Net Training"
    )
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
    # Find data
    # --------------------------------------------------------

    image_files, label_files = (
        find_pairs()
    )

    print(
        f"Total chips: "
        f"{len(image_files)}"
    )

    # --------------------------------------------------------
    # Create full dataset
    # --------------------------------------------------------

    full_dataset = SatelliteDataset(
        image_files,
        label_files,
        augment=False
    )

    # --------------------------------------------------------
    # 80/20 split
    # --------------------------------------------------------

    validation_size = int(
        len(full_dataset)
        * VALIDATION_SPLIT
    )

    training_size = (
        len(full_dataset)
        - validation_size
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    train_subset, val_subset = (
        random_split(
            full_dataset,
            [
                training_size,
                validation_size
            ],
            generator=generator
        )
    )

    # --------------------------------------------------------
    # Separate augmented training dataset
    # --------------------------------------------------------

    train_dataset = SatelliteDataset(
        [
            image_files[i]
            for i in train_subset.indices
        ],
        [
            label_files[i]
            for i in train_subset.indices
        ],
        augment=True
    )

    val_dataset = SatelliteDataset(
        [
            image_files[i]
            for i in val_subset.indices
        ],
        [
            label_files[i]
            for i in val_subset.indices
        ],
        augment=False
    )

    print(
        f"Training chips: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation chips: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # U-Net
    # --------------------------------------------------------

    print(
        "\nCreating U-Net..."
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES
    )

    model = model.to(
        device
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{total_params:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = -1.0

    best_epoch = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\nStarting training...")
    print(
        f"Epochs: {EPOCHS}"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss, val_acc = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        scheduler.step()

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "train_acc"
        ].append(
            train_acc
        )

        history[
            "val_acc"
        ].append(
            val_acc
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                MODEL_DIR
                / "unet_best.pth"
            )

            print(
                "  -> Best model saved."
            )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_DIR
        / "unet_final.pth"
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["best_val_acc"] = (
        best_val_acc
    )

    history["best_epoch"] = (
        best_epoch
    )

    history_file = (
        EVALUATION_DIR
        / "unet_history.json"
    )

    with open(
        history_file,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    save_training_curves(
        history
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"\nBest model:"
        f"\n{MODEL_DIR / 'unet_best.pth'}"
    )

    print(
        f"\nFinal model:"
        f"\n{MODEL_DIR / 'unet_final.pth'}"
    )

    print(
        f"\nTraining history:"
        f"\n{history_file}"
    )

    print(
        f"\nTraining curves:"
        f"\n{PLOT_DIR}"
    )


if __name__ == "__main__":
    main()