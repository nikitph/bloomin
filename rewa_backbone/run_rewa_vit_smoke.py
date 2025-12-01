# run_rewa_vit_smoke.py
import torch
from rewa_backbone.vision_backbone import REWAVisionBackbone

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # instantiate small model for smoke test
    model = REWAVisionBackbone(
        img_size=128,
        patch_size=16,
        in_chans=3,
        embed_dim=128,  # small for test
        depth=2,
        heads=4,
        mlp_ratio=2
    ).to(device)

    model.eval()

    # Single random image (B, C, H, W)
    x = torch.randn(1, 3, 128, 128, device=device)

    with torch.no_grad():
        out = model(x)   # returns CLS embedding
    print("Output shape:", out.shape)   # (1, embed_dim)

if __name__ == "__main__":
    main()
