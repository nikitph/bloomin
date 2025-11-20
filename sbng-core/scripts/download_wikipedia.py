#!/usr/bin/env python3
"""
Download Wikipedia dataset using the new datasets API.
"""

from datasets import load_dataset
import json
from pathlib import Path

def main():
    print("Downloading Wikipedia dataset from HuggingFace...")
    print("Using 'wikimedia/wikipedia' dataset (20231101.en subset)...")
    
    # Load English Wikipedia (smaller subset)
    # Using the new dataset path
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True  # Stream to avoid downloading everything
    )
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Save 10k articles
    output_path = output_dir / "wikipedia_10k.jsonl"
    print(f"Saving first 10,000 articles to {output_path}...")
    
    with open(output_path, 'w') as f:
        for i, item in enumerate(dataset):
            if i >= 10000:
                break
            
            if i % 1000 == 0:
                print(f"  Processed {i} articles...")
            
            # Extract title and text
            doc = {
                "id": f"wiki_{i}",
                "text": f"{item['title']}. {item['text']}"
            }
            f.write(json.dumps(doc) + '\n')
    
    print(f"\nSaved 10,000 Wikipedia articles to {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
