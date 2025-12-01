# train_rewa_vit_tiny.py
import torch, time
from torch import nn, optim
from rewa_backbone.vision_backbone import REWAVisionBackbone

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def train(steps=20, batch_size=4):
    print(f"Training on {device}...")
    model = REWAVisionBackbone(img_size=128, patch_size=16, embed_dim=128, depth=2, heads=4).to(device)
    head = nn.Linear(128, 10).to(device)  # toy classifier

    optimizer = optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        imgs = torch.randn(batch_size, 3, 128, 128, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)

        model.train()
        emb = model(imgs)   # (B, embed_dim)
        logits = head(emb)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"step {step} loss {loss.item():.4f}")

if __name__ == "__main__":
    train(steps=20, batch_size=4)
