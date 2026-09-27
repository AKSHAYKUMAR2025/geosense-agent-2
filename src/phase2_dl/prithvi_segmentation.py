import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from terratorch.registry import BACKBONE_REGISTRY


# ============================================================
# GeoSense Phase 2
# Exercise 4 - Prithvi Foundation Model
# Segmentation Model Setup
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

CHECKPOINT = Path(
    "models/prithvi_300m/Prithvi_EO_V2_300M.pt"
)

CHIP_PATH = Path(
    "data/processed/unet/chips/image_0000.npy"
)


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

NUM_CLASSES = 5

# Prithvi-EO-2.0-300M embedding dimension
EMBED_DIM = 1024

# Prithvi uses 16x16 spatial patches for 224x224 images
PATCH_GRID = 14

# Number of trainable final transformer blocks
TRAINABLE_BLOCKS = [22, 23]


# ============================================================
# Prithvi + Segmentation Head
# ============================================================

class PrithviSegmentationModel(nn.Module):

    def __init__(self):

        super().__init__()

        print(
            "Loading Prithvi-EO-2.0-300M..."
        )

        # ----------------------------------------------------
        # Load the actual Prithvi model
        # ----------------------------------------------------

        self.encoder = (
            BACKBONE_REGISTRY.build(
                "prithvi_eo_v2_300",
                pretrained=True,
                pretrained_cfg={
                    "checkpoint": str(
                        CHECKPOINT
                    )
                }
            )
        )

        # ----------------------------------------------------
        # Freeze the complete encoder first
        # ----------------------------------------------------

        for parameter in (
            self.encoder.parameters()
        ):

            parameter.requires_grad = False

        # ----------------------------------------------------
        # Architecture adaptation
        #
        # The actual Prithvi implementation does NOT contain
        # "layer1" and "layer2".
        #
        # It contains:
        #
        # blocks.0 ... blocks.23
        #
        # Therefore, for this implementation we fine-tune
        # the final two transformer blocks.
        # ----------------------------------------------------

        for block_index in TRAINABLE_BLOCKS:

            for parameter in (
                self.encoder
                .blocks[block_index]
                .parameters()
            ):

                parameter.requires_grad = True

        # ----------------------------------------------------
        # Segmentation head
        #
        # Input:
        #   1024-channel Prithvi features
        #
        # Output:
        #   5 land-cover classes
        # ----------------------------------------------------

        self.segmentation_head = nn.Sequential(

            nn.Conv2d(
                in_channels=EMBED_DIM,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                in_channels=256,
                out_channels=NUM_CLASSES,
                kernel_size=1
            )
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x
    ):

        # ----------------------------------------------------
        # Input expected from our chips:
        #
        # [B, 6, 224, 224]
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "Expected input shape "
                "[B, 6, 224, 224], "
                f"got {x.shape}"
            )

        if x.shape[1] != 6:

            raise ValueError(
                "Expected 6 input channels, "
                f"got {x.shape[1]}"
            )

        # ----------------------------------------------------
        # Prithvi expects a temporal dimension:
        #
        # [B, C, T, H, W]
        #
        # Our current chips contain one temporal observation,
        # so add T=1.
        # ----------------------------------------------------

        x = x.unsqueeze(2)

        # Shape:
        #
        # [B, 6, 1, 224, 224]
        #

        # ----------------------------------------------------
        # Get Prithvi encoder features
        # ----------------------------------------------------

        features = (
            self.encoder.forward_features(
                x
            )
        )

        # ----------------------------------------------------
        # The actual model returns a collection of features
        # from the transformer blocks.
        #
        # Use the final block output.
        # ----------------------------------------------------

        if isinstance(
            features,
            (list, tuple)
        ):

            features = features[-1]

        # ----------------------------------------------------
        # Actual observed output:
        #
        # [B, 197, 1024]
        #
        # 197 =
        #     1 CLS token
        #     +
        #     196 spatial tokens
        #
        # 196 = 14 x 14
        # ----------------------------------------------------

        if features.ndim != 3:

            raise RuntimeError(
                "Unexpected Prithvi feature "
                f"dimensions: {features.shape}"
            )

        batch_size = (
            features.shape[0]
        )

        token_count = (
            features.shape[1]
        )

        embedding_dim = (
            features.shape[2]
        )

        print_once = False

        # ----------------------------------------------------
        # Handle 197-token output
        # ----------------------------------------------------

        if token_count == 197:

            # Remove CLS token
            features = (
                features[:, 1:, :]
            )

        elif token_count == 196:

            # Already spatial-only
            pass

        else:

            raise RuntimeError(
                "Unexpected Prithvi "
                f"feature shape: {features.shape}. "
                "Expected [B,197,1024] or "
                "[B,196,1024]."
            )

        # ----------------------------------------------------
        # Verify embedding dimension
        # ----------------------------------------------------

        if embedding_dim != EMBED_DIM:

            raise RuntimeError(
                "Unexpected Prithvi embedding "
                f"dimension: {embedding_dim}. "
                f"Expected {EMBED_DIM}."
            )

        # ----------------------------------------------------
        # Now:
        #
        # [B, 196, 1024]
        #
        # Convert to:
        #
        # [B, 14, 14, 1024]
        # ----------------------------------------------------

        features = features.reshape(
            batch_size,
            PATCH_GRID,
            PATCH_GRID,
            EMBED_DIM
        )

        # ----------------------------------------------------
        # Convert:
        #
        # [B, H, W, C]
        #
        # to:
        #
        # [B, C, H, W]
        # ----------------------------------------------------

        features = (
            features.permute(
                0,
                3,
                1,
                2
            )
            .contiguous()
        )

        # Shape:
        #
        # [B, 1024, 14, 14]
        #

        # ----------------------------------------------------
        # Segmentation head
        # ----------------------------------------------------

        output = (
            self.segmentation_head(
                features
            )
        )

        # Shape:
        #
        # [B, 5, 14, 14]
        #

        # ----------------------------------------------------
        # Upsample to original chip size
        # ----------------------------------------------------

        output = F.interpolate(
            output,
            size=(
                224,
                224
            ),
            mode="bilinear",
            align_corners=False
        )

        # Final:
        #
        # [B, 5, 224, 224]
        #

        return output


# ============================================================
# Parameter statistics
# ============================================================

def count_parameters(
    model
):

    total = sum(
        parameter.numel()
        for parameter in (
            model.parameters()
        )
    )

    trainable = sum(
        parameter.numel()
        for parameter in (
            model.parameters()
        )
        if parameter.requires_grad
    )

    frozen = (
        total
        - trainable
    )

    return (
        total,
        trainable,
        frozen
    )


# ============================================================
# Main sanity check
# ============================================================

def main():

    print("=" * 70)

    print(
        "GeoSense Phase 2 - "
        "Prithvi Segmentation Setup"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "Prithvi checkpoint not found:\n"
            f"{CHECKPOINT}"
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
    # Create model
    # --------------------------------------------------------

    model = (
        PrithviSegmentationModel()
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Parameter counts
    # --------------------------------------------------------

    (
        total,
        trainable,
        frozen
    ) = count_parameters(
        model
    )

    print(
        "\nParameter summary"
    )

    print(
        "-" * 70
    )

    print(
        f"Total parameters:     "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Frozen parameters:    "
        f"{frozen:,}"
    )

    # --------------------------------------------------------
    # Show trainable blocks
    # --------------------------------------------------------

    print(
        "\nTrainable components"
    )

    print(
        "-" * 70
    )

    for name, parameter in (
        model.named_parameters()
    ):

        if parameter.requires_grad:

            print(
                name
            )

    # --------------------------------------------------------
    # Check chip
    # --------------------------------------------------------

    if not CHIP_PATH.exists():

        raise FileNotFoundError(
            "Training chip not found:\n"
            f"{CHIP_PATH}"
        )

    print(
        "\nLoading test chip..."
    )

    chip = np.load(
        CHIP_PATH
    ).astype(
        np.float32
    )

    print(
        f"Chip shape: {chip.shape}"
    )

    # --------------------------------------------------------
    # Verify chip
    # --------------------------------------------------------

    if chip.shape != (
        6,
        224,
        224
    ):

        raise RuntimeError(
            "Unexpected chip shape: "
            f"{chip.shape}"
        )

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    input_tensor = (
        torch.from_numpy(
            chip
        )
        .unsqueeze(0)
        .to(device)
    )

    print(
        "\nInput tensor"
    )

    print(
        "-" * 70
    )

    print(
        f"Shape: {input_tensor.shape}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print(
        "\nRunning Prithvi forward pass..."
    )

    model.eval()

    with torch.no_grad():

        output = model(
            input_tensor
        )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    expected_shape = (
        1,
        5,
        224,
        224
    )

    if tuple(
        output.shape
    ) != expected_shape:

        raise RuntimeError(
            "Unexpected output shape: "
            f"{output.shape}. "
            f"Expected {expected_shape}."
        )

    # --------------------------------------------------------
    # Check numerical output
    # --------------------------------------------------------

    if not torch.isfinite(
        output
    ).all():

        raise RuntimeError(
            "Model produced NaN "
            "or infinite values."
        )

    print(
        "Output numerical check: PASSED"
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print(
        "\nForward pass: PASSED"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PRITHVI SEGMENTATION SETUP PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()