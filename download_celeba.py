#!/usr/bin/env python3
"""Download CelebA images from HuggingFace"""

from datasets import load_dataset
from pathlib import Path
import sys

def download_celeba(output_dir='celeba_data', n_images=100):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Downloading {n_images} CelebA images from HuggingFace...")
    print(f"Output directory: {output_path.absolute()}")
    
    try:
        # Load dataset in streaming mode (faster, doesn't download everything)
        dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)
        
        count = 0
        for i, sample in enumerate(dataset):
            if count >= n_images:
                break
            
            img = sample['image']
            img_path = output_path / f'{count:05d}.jpg'
            img.save(img_path)
            count += 1
            
            if count % 10 == 0:
                print(f"  Downloaded {count}/{n_images} images...")
        
        print(f"\n✅ Successfully downloaded {count} images to {output_path}/")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try: pip install --upgrade datasets")
        print("3. Or manually download from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        return False

if __name__ == "__main__":
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    success = download_celeba(n_images=n_images)
    sys.exit(0 if success else 1)
