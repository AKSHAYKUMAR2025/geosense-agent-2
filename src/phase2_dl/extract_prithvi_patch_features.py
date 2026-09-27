from pathlib import Path

import numpy as np
import rasterio
import torch


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "prithvi_300m"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "prithvi"

CHECKPOINT = MODEL_DIR / "Prithvi_EO_V2_300M.pt"
CONFIG_FILE = MODEL_DIR / "config.json"


# ---------------------------------------------------------
# 2. Import the official Prithvi model
# ---------------------------------------------------------

import sys

sys.path.insert(0, str(MODEL_DIR))

from prithvi_mae import PrithviMAE


# ---------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------

PATCH_SIZE = 224

MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
    dtype=np.float32,
)

STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
    dtype=np.float32,
)

TEMPORAL_FILES = [
    DATA_DIR / "chennai_prithvi_t1.tif",
    DATA_DIR / "chennai_prithvi_t2.tif",
    DATA_DIR / "chennai_prithvi_t3.tif",
    DATA_DIR / "chennai_prithvi_t4.tif",
]


# ---------------------------------------------------------
# 4. Load one temporal raster
# ---------------------------------------------------------

def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)

    print(f"Loaded: {path.name}")
    print(f"Shape : {data.shape}")

    return data


# ---------------------------------------------------------
# 5. Normalize using official Prithvi statistics
# ---------------------------------------------------------

def normalize(data):
    mean = MEAN[:, None, None]
    std = STD[:, None, None]

    return (data - mean) / std


# ---------------------------------------------------------
# 6. Create one Prithvi patch
# ---------------------------------------------------------

def make_patch(rasters, y, x):
    temporal_patches = []

    for raster in rasters:

        patch = raster[
            :,
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE,
        ]

        # Reflective padding if patch reaches an edge
        pad_h = PATCH_SIZE - patch.shape[1]
        pad_w = PATCH_SIZE - patch.shape[2]

        if pad_h > 0 or pad_w > 0:
            patch = np.pad(
                patch,
                (
                    (0, 0),
                    (0, pad_h),
                    (0, pad_w),
                ),
                mode="reflect",
            )

        patch = normalize(patch)

        temporal_patches.append(patch)

    # (time, channels, height, width)
    data = np.stack(temporal_patches, axis=0)

    # Prithvi expects:
    # (batch, channels, time, height, width)
    data = np.transpose(data, (1, 0, 2, 3))

    data = torch.from_numpy(data).unsqueeze(0)

    return data


# ---------------------------------------------------------
# 7. Load Prithvi
# ---------------------------------------------------------

def load_model():

    print()
    print("Creating Prithvi model...")

    model = PrithviMAE(
        img_size=224,
        patch_size=(1, 16, 16),
        in_chans=6,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=torch.nn.LayerNorm,
        norm_pix_loss=False,
        mask_ratio=0.75,
    )

    print("Model architecture created.")

    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Official inference removes fixed positional embedding
    state_dict.pop("pos_embed", None)

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    model.eval()

    print("Prithvi model loaded successfully.")

    return model


# ---------------------------------------------------------
# 8. Extract embedding from one patch
# ---------------------------------------------------------

def extract_embedding(model, patch):

    with torch.no_grad():

        features = model.forward_features(
            patch,
            temporal_coords=None,
            location_coords=None,
        )

        # Final transformer layer
        final_features = features[-1]

        # Remove CLS token
        tokens = final_features[:, 1:, :]

        # 784 tokens = 4 temporal frames × 196 spatial tokens
        batch_size = tokens.shape[0]

        tokens = tokens.reshape(
            batch_size,
            4,
            196,
            1024,
        )

        # Average spatial tokens
        temporal_embeddings = tokens.mean(dim=2)

        # Flatten the four temporal embeddings
        embedding = temporal_embeddings.reshape(
            batch_size,
            4 * 1024,
        )

    return embedding


# ---------------------------------------------------------
# 9. Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("GeoSense Phase 2 - Prithvi Patch Feature Extraction")
    print("=" * 60)

    print()
    print("Device: CPU")

    # Load four temporal rasters
    rasters = [
        load_raster(path)
        for path in TEMPORAL_FILES
    ]

    height = rasters[0].shape[1]
    width = rasters[0].shape[2]

    print()
    print("Scene dimensions:")
    print("Width :", width)
    print("Height:", height)

    # 652 x 446 scene
    # 224 x 224 Prithvi patches
    x_positions = [0, 224, 448]
    y_positions = [0, 224]

    print()
    print("Patch grid:")
    print("Horizontal patches:", len(x_positions))
    print("Vertical patches  :", len(y_positions))
    print("Total patches     :", len(x_positions) * len(y_positions))

    # Load model
    model = load_model()

    embeddings = []

    patch_id = 0

    for y in y_positions:

        for x in x_positions:

            patch_id += 1

            print()
            print("-" * 60)
            print(f"Processing patch {patch_id}/6")
            print(f"Pixel position: x={x}, y={y}")

            patch = make_patch(
                rasters,
                y,
                x,
            )

            print("Input shape:", tuple(patch.shape))

            embedding = extract_embedding(
                model,
                patch,
            )

            print(
                "Embedding shape:",
                tuple(embedding.shape),
            )

            embeddings.append(
                embedding.squeeze(0).numpy()
            )

    embeddings = np.stack(
        embeddings,
        axis=0,
    )

    print()
    print("=" * 60)
    print("PRITHVI PATCH EXTRACTION COMPLETE")
    print("=" * 60)

    print("Patch embeddings:", embeddings.shape)

    print()
    print("Expected:")
    print("6 patches × 4096 features")

    # Save
    output_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "prithvi_patch_embeddings.npy"
    )

    np.save(
        output_file,
        embeddings,
    )

    print()
    print("Saved:", output_file)


if __name__ == "__main__":
    main()