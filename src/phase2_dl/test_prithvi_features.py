from pathlib import Path
import sys

import numpy as np
import rasterio
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "prithvi_300m"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "prithvi"

sys.path.insert(0, str(MODEL_DIR))

from prithvi_mae import PrithviMAE


IMG_SIZE = 224

MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
    dtype=np.float32,
)

STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
    dtype=np.float32,
)


# ---------------------------------------------------------
# Load four temporal images
# ---------------------------------------------------------
files = [
    DATA_DIR / "chennai_prithvi_t1.tif",
    DATA_DIR / "chennai_prithvi_t2.tif",
    DATA_DIR / "chennai_prithvi_t3.tif",
    DATA_DIR / "chennai_prithvi_t4.tif",
]

images = []

for file in files:
    with rasterio.open(file) as src:
        image = src.read().astype(np.float32)

    images.append(image[:, :IMG_SIZE, :IMG_SIZE])


# ---------------------------------------------------------
# Build C,T,H,W
# ---------------------------------------------------------
data = np.stack(images, axis=0)
data = np.transpose(data, (1, 0, 2, 3))

data = (
    data - MEAN[:, None, None, None]
) / STD[:, None, None, None]

input_data = torch.from_numpy(
    data[None, ...]
).float()

print("Input shape:", tuple(input_data.shape))


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------
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
# Extract encoder features
# ---------------------------------------------------------
print("Extracting Prithvi encoder features...")
print("Please wait...")

with torch.no_grad():
    features = model.forward_features(
        input_data,
        None,
        None,
    )


# ---------------------------------------------------------
# Inspect result
# ---------------------------------------------------------
print()
print("========================================")
print("PRITHVI FEATURE EXTRACTION SUCCESSFUL")
print("========================================")

print("Python type:", type(features))

if isinstance(features, (list, tuple)):
    print("Number of feature outputs:", len(features))

    for i, feature in enumerate(features):
        print(
            f"Feature {i}:",
            type(feature),
            getattr(feature, "shape", None),
        )
else:
    print(
        "Feature shape:",
        getattr(features, "shape", None),
    )