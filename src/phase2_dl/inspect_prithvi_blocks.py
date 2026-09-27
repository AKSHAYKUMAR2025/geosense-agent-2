import torch
from pathlib import Path
from terratorch.registry import BACKBONE_REGISTRY


CHECKPOINT = Path(
    "models/prithvi_300m/Prithvi_EO_V2_300M.pt"
)


def main():

    print("=" * 70)
    print("Prithvi Transformer Block Inspection")
    print("=" * 70)

    model = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300",
        pretrained=True,
        pretrained_cfg={
            "checkpoint": str(CHECKPOINT)
        }
    )

    print("\nTop-level model modules:")
    print("-" * 70)

    for name, module in model.named_children():

        print(
            f"{name:20s} "
            f"{module.__class__.__name__}"
        )

    print("\nTransformer blocks:")
    print("-" * 70)

    if hasattr(model, "blocks"):

        print(
            f"Number of blocks: "
            f"{len(model.blocks)}"
        )

        for i, block in enumerate(
            model.blocks
        ):

            parameter_count = sum(
                p.numel()
                for p in block.parameters()
            )

            print(
                f"Block {i:02d}: "
                f"{block.__class__.__name__:20s} "
                f"parameters={parameter_count:,}"
            )

    else:

        print(
            "No 'blocks' attribute found."
        )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()