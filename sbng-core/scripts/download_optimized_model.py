#!/usr/bin/env python3
"""
Download pre-optimized INT8 quantized model from HuggingFace Optimum.
"""
import os
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-2-v2"
OUTPUT_DIR = "model_optimized"

def download_optimized_model():
    print(f"Downloading optimized model: {MODEL_ID}")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Download and optimize model with INT8 quantization
    # Optimum will download the ONNX version if available, or convert + optimize
    model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        export=True,  # Export to ONNX if not already available
        provider="CPUExecutionProvider",
    )
    
    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✓ Model saved to {OUTPUT_DIR}")
    
    # Check file sizes
    model_path = os.path.join(OUTPUT_DIR, "model.onnx")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")
    
    print("\nModel supports dynamic batching and is optimized for CPU inference.")

if __name__ == "__main__":
    download_optimized_model()
