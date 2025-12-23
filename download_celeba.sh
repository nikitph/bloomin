#!/bin/bash
# Download CelebA dataset images

echo "Downloading CelebA dataset..."

# Using Kaggle's CelebA dataset (easier access than Google Drive)
# First, we'll download a small subset using a public mirror

cd celeba_data

# Download from a public CelebA mirror (first 100 images)
# Note: The official CelebA requires registration, so we'll use a smaller public subset
wget -q --show-progress https://github.com/tkarras/progressive_growing_of_gans/raw/master/datasets/download_celeba.py -O download_celeba.py

# Alternative: Download from a public subset
echo "Downloading sample CelebA images from public source..."

# Create a simple Python script to download from HuggingFace
cat > download_hf.py << 'EOF'
from huggingface_hub import hf_hub_download
import os

# Download sample images from HuggingFace datasets
try:
    from datasets import load_dataset
    print("Loading CelebA dataset from HuggingFace...")
    dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)
    
    # Download first 100 images
    count = 0
    for i, sample in enumerate(dataset):
        if count >= 100:
            break
        img = sample['image']
        img.save(f'{count:05d}.jpg')
        count += 1
        if count % 10 == 0:
            print(f"Downloaded {count} images...")
    
    print(f"✅ Downloaded {count} images successfully!")
except Exception as e:
    print(f"Error: {e}")
    print("Please install datasets: pip install datasets")
EOF

python download_hf.py
cd ..
