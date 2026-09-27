from pathlib import Path
import sys

import numpy as np
import rasterio
import torch


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "prithvi_300m"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "prithvi"

sys.path.insert(0, str(MODEL_DIR))

from prithvi_mae import PrithviMAE


# ---------------------------------------------------------
# Prithvi configuration
# ---------------------------------------------------------
IMG_SIZE = 224
NUM_FRAMES = 4

MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
    dtype=np.float32,
)

STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
    dtype=np.float32,
)


# ---------------------------------------------------------
# Read four temporal images
# ---------------------------------------------------------
files = [
    DATA_DIR / "chennai_prithvi_t1.tif",
    DATA_DIR / "chennai_prithvi_t2.tif",
    DATA_DIR / "chennai_prithvi_t3.tif",
    DATA_DIR / "chennai_prithvi_t4.tif",
]

print("Loading four temporal images...")

images = []

for file in files:
    print("Reading:", file.name)

    with rasterio.open(file) as src:
        image = src.read().astype(np.float32)

    print("Shape:", image.shape)

    if image.shape[0] != 6:
        raise ValueError(
            f"{file.name} does not contain 6 bands."
        )

    images.append(image)


# ---------------------------------------------------------
# Crop first 224 x 224 pixels
# ---------------------------------------------------------
images = [
    image[:, :IMG_SIZE, :IMG_SIZE]
    for image in images
]


# ---------------------------------------------------------
# Stack temporal dimension
# ---------------------------------------------------------
# Current:
#   each image = C,H,W
#
# Stack:
#   T,C,H,W
#
# Then transpose:
#   C,T,H,W
# ---------------------------------------------------------
data = np.stack(images, axis=0)

data = np.transpose(data, (1, 0, 2, 3))


# ---------------------------------------------------------
# Normalize using official Prithvi statistics
# ---------------------------------------------------------
data = (
    data - MEAN[:, None, None, None]
) / STD[:, None, None, None]


# Add batch dimension:
# C,T,H,W -> B,C,T,H,W
input_data = torch.from_numpy(
    data[None, ...]
).float()


print()
print("Input tensor:")
print("Shape:", tuple(input_data.shape))
print("Expected: (1, 6, 4, 224, 224)")


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------
print()
print("Creating Prithvi model...")

model = PrithviMAE(
    img_size=224,
    num_frames=4,
    patch_size=(1, 16, 16),
    in_chans=6,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    coords_encoding=[],
    coords_scale_learn=False,
)


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------
checkpoint = MODEL_DIR / "Prithvi_EO_V2_300M.pt"

print("Loading checkpoint...")

state_dict = torch.load(
    checkpoint,
    map_location="cpu",
    weights_only=True,
)

for key in list(state_dict.keys()):
    if "pos_embed" in key:
        del state_dict[key]

model.load_state_dict(
    state_dict,
    strict=False,
)

model.eval()


# ---------------------------------------------------------
# Run one inference
# ---------------------------------------------------------
print()
print("Running Prithvi inference on one 224x224 patch...")
print("Device: CPU")
print("Please wait...")


with torch.no_grad():

    latent, prediction, mask = model(
        input_data,
        None,
        None,
        0.75,
    )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
print()
print("========================================")
print("PRITHVI PATCH INFERENCE SUCCESSFUL")
print("========================================")

print("Input shape :", tuple(input_data.shape))
print("Latent shape:", tuple(latent.shape))
print("Prediction  :", tuple(prediction.shape))
print("Mask shape  :", tuple(mask.shape))