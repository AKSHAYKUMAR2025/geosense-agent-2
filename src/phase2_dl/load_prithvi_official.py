from pathlib import Path
import sys
import torch


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "prithvi_300m"
CHECKPOINT = MODEL_DIR / "Prithvi_EO_V2_300M.pt"


# ---------------------------------------------------------
# Import official Prithvi model
# ---------------------------------------------------------
sys.path.insert(0, str(MODEL_DIR))

from prithvi_mae import PrithviMAE


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------
config = {
    "img_size": 224,
    "num_frames": 4,
    "patch_size": (1, 16, 16),
    "in_chans": 6,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "decoder_embed_dim": 512,
    "decoder_depth": 8,
    "decoder_num_heads": 16,
    "mlp_ratio": 4,
    "coords_encoding": [],
    "coords_scale_learn": False,
}


print("Creating Prithvi model...")

model = PrithviMAE(**config)

print("Model architecture created.")

print("Loading checkpoint...")

device = torch.device("cpu")

state_dict = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True
)

# Official inference code removes fixed positional embedding.
for key in list(state_dict.keys()):
    if "pos_embed" in key:
        del state_dict[key]

model.load_state_dict(
    state_dict,
    strict=False
)

model.to(device)
model.eval()

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print()
print("========================================")
print("PRITHVI MODEL LOADED SUCCESSFULLY")
print("========================================")
print("Parameters:", f"{total_params:,}")
print("Device:", device)
print("Checkpoint:", CHECKPOINT)