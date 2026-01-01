import torch
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
import os
from PIL import Image

# Configuration
DATA_DIR = './data'
OUTPUT_FILE = './data/cifar_pei.json'
N_ITEMS = 2000 # Limit to 2k for speed, or 5k

def extract_color_metadata(image_tensor):
    # image_tensor: [3, 32, 32] float 0..1
    # Simple dominant color: Mean of R, G, B
    means = torch.mean(image_tensor, dim=[1, 2]) # [3]
    r, g, b = means[0].item(), means[1].item(), means[2].item()
    
    # Simple quantization
    # 0: Red, 1: Green, 2: Blue, 3: Dark, 4: Light
    if r > g and r > b and r > 0.4: return 0 
    if g > r and g > b and g > 0.4: return 1
    if b > r and b > g and b > 0.4: return 2
    if r + g + b < 1.0: return 3 # Dark
    return 4 # Light/Mixed

def main():
    print("Preparing CIFAR-10 Data...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load Model (Pre-trained ResNet18)
    # We remove the FC layer to get 512D embeddings
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity() # Replace classification head
    model.eval()

    # 2. Load Data
    transform = transforms.Compose([
        transforms.Resize(224), # ResNet expects 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # We need raw images for metadata, transformed for vectors
    dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    raw_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    
    data_out = []
    
    print(f"Processing {N_ITEMS} images...")
    with torch.no_grad():
        for i in range(N_ITEMS):
            img_t, label = dataset[i]
            img_raw, _ = raw_dataset[i]
            
            # Embed
            # unsqueeze(0) -> [1, 3, 224, 224]
            emb = model(img_t.unsqueeze(0)).squeeze(0).numpy().tolist() # [512]
            
            # Metadata
            color = extract_color_metadata(img_raw)
            # Aspect is always 1.0 for CIFAR (32x32), but let's pretend some variation or just hardcode
            aspect = 1.0 
            
            # Coarse: First 32 dims of ResNet? 
            # ResNet features aren't ordered by frequency. 
            # A random projection or just taking the first 32 is a valid "Weak Signal".
            coarse = emb[:32]
            
            item = {
                "id": i,
                "vector": emb,
                "metadata": {
                    "aspect_ratio": aspect,
                    "color": color,
                    "coarse_emb": coarse,
                    "label": label 
                }
            }
            data_out.append(item)
            if i % 100 == 0: print(f"Processed {i}/{N_ITEMS}")

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data_out, f)
    print("Done.")

if __name__ == "__main__":
    main()
