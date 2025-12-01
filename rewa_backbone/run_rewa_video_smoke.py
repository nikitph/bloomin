# run_rewa_video_smoke.py
import torch
from rewa_backbone.video_backbone import REWAVideoBackbone

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # instantiate small model for smoke test
    model = REWAVideoBackbone(
        in_channels=3,
        patch_size=(2, 16, 16),
        embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        num_frames=8,
        img_size=128,
        num_classes=10
    ).to(device)

    model.eval()

    # Single random video (B, C, T, H, W)
    x = torch.randn(1, 3, 8, 128, 128, device=device)

    with torch.no_grad():
        out = model(x)   # returns logits
    print("Output shape:", out.shape)   # (1, num_classes)

if __name__ == "__main__":
    main()
