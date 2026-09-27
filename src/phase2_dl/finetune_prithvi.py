import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from prithvi_segmentation import PrithviSegmentationModel


# ============================================================
# GeoSense Phase 2
# Exercise 4 - Prithvi Foundation Model Fine-Tuning
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

# IMPORTANT:
# Your actual project stores both image and label chips here.

CHIP_DIR = Path(
    "data/processed/unet/chips"
)

MODEL_DIR = Path(
    "models/saved"
)

EVALUATION_DIR = Path(
    "models/evaluation"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Model / training configuration
# ============================================================

NUM_CLASSES = 5

# Actual Prithvi architecture:
# blocks.0 ... blocks.23
#
# We freeze blocks 0-21 and fine-tune
# the final two transformer blocks.

TRAINABLE_BLOCKS = [
    22,
    23
]

BATCH_SIZE = 1

# Short CPU validation run.
EPOCHS = 3

VAL_RATIO = 0.20

LEARNING_RATE_BACKBONE = 1e-5

LEARNING_RATE_HEAD = 1e-4

WEIGHT_DECAY = 1e-4

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

        image_path = (
            self.image_files[index]
        )

        label_path = (
            self.label_files[index]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = np.load(
            image_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Load label
        # ----------------------------------------------------

        label = np.load(
            label_path
        ).astype(
            np.int64
        )

        # ----------------------------------------------------
        # Check image shape
        # ----------------------------------------------------

        if image.shape != (
            6,
            224,
            224
        ):

            raise ValueError(
                f"Unexpected image shape "
                f"{image.shape} in "
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Check label shape
        # ----------------------------------------------------

        if label.shape != (
            224,
            224
        ):

            raise ValueError(
                f"Unexpected label shape "
                f"{label.shape} in "
                f"{label_path}"
            )

        # ----------------------------------------------------
        # Replace invalid values
        # ----------------------------------------------------

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        )

        label = torch.from_numpy(
            label
        )

        return image, label


# ============================================================
# Find matching image / label pairs
# ============================================================

def find_chip_pairs():

    image_files = sorted(
        CHIP_DIR.glob(
            "image_*.npy"
        )
    )

    if len(image_files) == 0:

        raise FileNotFoundError(
            "No image chips found in:\n"
            f"{CHIP_DIR}"
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

        if not label_path.exists():

            print(
                f"WARNING: Missing label for "
                f"{image_path.name}"
            )

            continue

        valid_images.append(
            image_path
        )

        valid_labels.append(
            label_path
        )

    if len(valid_images) == 0:

        raise FileNotFoundError(
            "No matching image/label pairs found."
        )

    return (
        valid_images,
        valid_labels
    )


# ============================================================
# Pixel accuracy
# ============================================================

def pixel_accuracy(
    outputs,
    targets
):

    predictions = (
        outputs.argmax(
            dim=1
        )
    )

    correct = (
        predictions == targets
    ).sum().item()

    total = targets.numel()

    if total == 0:

        return 0.0

    return (
        correct / total
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    total_accuracy = 0.0

    batches = 0

    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images
        )

        # ----------------------------------------------------
        # Cross entropy loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
        )

        total_accuracy += (
            pixel_accuracy(
                outputs,
                labels
            )
        )

        batches += 1

    return (
        total_loss / batches,
        total_accuracy / batches
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

    total_loss = 0.0

    total_accuracy = 0.0

    batches = 0

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

            total_loss += (
                loss.item()
            )

            total_accuracy += (
                pixel_accuracy(
                    outputs,
                    labels
                )
            )

            batches += 1

    return (
        total_loss / batches,
        total_accuracy / batches
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "GeoSense Phase 2 - "
        "Prithvi Fine-Tuning"
    )

    print("=" * 70)

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

    if device.type == "cpu":

        print(
            "\nWARNING:"
        )

        print(
            "CPU training will be slow."
        )

        print(
            f"Test run configured for "
            f"{EPOCHS} epochs."
        )

    # --------------------------------------------------------
    # Find chips
    # --------------------------------------------------------

    print(
        "\nSearching for training chips..."
    )

    (
        image_files,
        label_files
    ) = find_chip_pairs()

    print(
        f"Image chips: {len(image_files)}"
    )

    print(
        f"Label chips: {len(label_files)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SatelliteDataset(
        image_files,
        label_files
    )

    total_samples = len(
        dataset
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    validation_size = max(
        1,
        int(
            total_samples
            * VAL_RATIO
        )
    )

    training_size = (
        total_samples
        - validation_size
    )

    if training_size < 1:

        raise RuntimeError(
            "Not enough samples for "
            "training and validation."
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
            training_size,
            validation_size
        ],
        generator=generator
    )

    print(
        f"\nTraining chips: "
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
    # Create Prithvi segmentation model
    # --------------------------------------------------------

    print(
        "\nCreating Prithvi model..."
    )

    model = (
        PrithviSegmentationModel()
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Collect trainable parameters
    # --------------------------------------------------------

    backbone_parameters = []

    head_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:

            continue

        if name.startswith(
            "encoder."
        ):

            backbone_parameters.append(
                parameter
            )

        elif name.startswith(
            "segmentation_head."
        ):

            head_parameters.append(
                parameter
            )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if len(
        backbone_parameters
    ) == 0:

        raise RuntimeError(
            "No trainable backbone "
            "parameters found."
        )

    if len(
        head_parameters
    ) == 0:

        raise RuntimeError(
            "No trainable segmentation "
            "head parameters found."
        )

    # --------------------------------------------------------
    # Print training configuration
    # --------------------------------------------------------

    print(
        "\nFine-tuning configuration"
    )

    print(
        "-" * 70
    )

    print(
        f"Trainable blocks: "
        f"{TRAINABLE_BLOCKS}"
    )

    print(
        f"Backbone learning rate: "
        f"{LEARNING_RATE_BACKBONE}"
    )

    print(
        f"Head learning rate: "
        f"{LEARNING_RATE_HEAD}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    # --------------------------------------------------------
    # Optimizer
    #
    # Manual specifies AdamW with:
    #
    # Backbone = 1e-5
    # Head     = 1e-4
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    backbone_parameters,
                "lr":
                    LEARNING_RATE_BACKBONE
            },
            {
                "params":
                    head_parameters,
                "lr":
                    LEARNING_RATE_HEAD
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=2,
            gamma=0.5
        )
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

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nStarting training..."
    )

    print(
        "-" * 70
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_acc = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_acc * 100:.2f}%"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_acc * 100:.2f}%"
        )

        print(
            f"Backbone LR: "
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        print(
            f"Head LR: "
            f"{optimizer.param_groups[1]['lr']:.2e}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            checkpoint_path = (
                MODEL_DIR
                / "prithvi_finetuned.pth"
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_loss":
                        val_loss,

                    "val_acc":
                        val_acc,

                    "num_classes":
                        NUM_CLASSES,

                    "trainable_blocks":
                        TRAINABLE_BLOCKS,

                    "backbone_lr":
                        LEARNING_RATE_BACKBONE,

                    "head_lr":
                        LEARNING_RATE_HEAD,

                    "input_channels":
                        6,

                    "chip_size":
                        224
                },
                checkpoint_path
            )

            print(
                "\nBest model saved:"
            )

            print(
                checkpoint_path
            )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_path = (
        MODEL_DIR
        / "prithvi_finetuned_final.pth"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "num_classes":
                NUM_CLASSES,

            "trainable_blocks":
                TRAINABLE_BLOCKS
        },
        final_path
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        EVALUATION_DIR
        / "prithvi_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    if best_epoch > 0:

        print(
            f"Best validation accuracy: "
            f"{history['val_acc'][best_epoch - 1] * 100:.2f}%"
        )

    print(
        "\nSaved files:"
    )

    print(
        f"Best model:\n"
        f"{MODEL_DIR / 'prithvi_finetuned.pth'}"
    )

    print(
        f"\nFinal model:\n"
        f"{final_path}"
    )

    print(
        f"\nTraining history:\n"
        f"{history_path}"
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()