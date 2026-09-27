import torch
from terratorch.registry import BACKBONE_REGISTRY


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_NAME = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"

DEVICE = torch.device("cpu")


# ============================================================
# 2. START
# ============================================================

print("========================================")
print("PRITHVI-EO-2.0 MODEL TEST")
print("========================================")

print(f"Model : {MODEL_NAME}")
print(f"Device: {DEVICE}")

print("\nLoading Prithvi model...")
print("The first run may download approximately 1.34 GB.")
print("Please wait...")


# ============================================================
# 3. LOAD MODEL
# ============================================================

model = BACKBONE_REGISTRY.build(
    MODEL_NAME
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# 4. MODEL INFORMATION
# ============================================================

print("\nModel loaded successfully.")

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Parameters: "
    f"{parameter_count:,}"
)

print(
    f"Parameters (millions): "
    f"{parameter_count / 1_000_000:.1f}M"
)


# ============================================================
# 5. CREATE TEST INPUT
# ============================================================

# Prithvi-EO-2.0 expects:
#
#   6 spectral bands
#   4 temporal frames
#   224 x 224 spatial dimensions
#
# Tensor shape:
#
#   [batch, channels, time, height, width]

test_input = torch.randn(
    1,
    6,
    4,
    224,
    224,
    device=DEVICE
)

print("\nTest input created.")
print(
    f"Input shape: {tuple(test_input.shape)}"
)


# ============================================================
# 6. RUN TEST INFERENCE
# ============================================================

print("\nRunning CPU inference...")
print("This may take a while on CPU.")

with torch.no_grad():

    try:

        output = model(test_input)

        print("\nInference completed.")

        if isinstance(output, torch.Tensor):

            print(
                f"Output type : Tensor"
            )

            print(
                f"Output shape: {tuple(output.shape)}"
            )

        else:

            print(
                f"Output type: {type(output)}"
            )

            print(
                "The model returned a non-tensor "
                "output structure."
            )

    except Exception as e:

        print("\nModel loaded, but test inference failed.")
        print(
            f"Error type: {type(e).__name__}"
        )
        print(
            f"Error     : {e}"
        )
        raise


# ============================================================
# 7. FINAL RESULT
# ============================================================

print("\n========================================")
print("PRITHVI MODEL TEST COMPLETE")
print("========================================")

print("Prithvi model: OK")
print("TerraTorch   : OK")
print("PyTorch      : OK")
print("Device       : CPU")