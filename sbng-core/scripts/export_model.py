import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-2-v2"
OUTPUT_DIR = "model_quantized"

def export_model():
    print(f"Downloading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Tokenizer saved to {OUTPUT_DIR}")

    # Create dummy input with FIXED batch size for quantization compatibility
    # We'll use batch_size=1 and pad in Rust code
    dummy_input = tokenizer("query", "document", return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    
    # Export to ONNX with STATIC shapes (required for quantization)
    onnx_path = os.path.join(OUTPUT_DIR, "model.onnx")
    print("Exporting to ONNX with static shapes (batch=1, seq_len=128)...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input["token_type_ids"]),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        # NO dynamic_axes - use static shapes for quantization
        opset_version=14,  # Use older opset for better compatibility
    )
    print(f"Model exported to {onnx_path}")

    # Quantize with per-channel for better accuracy
    quantized_path = os.path.join(OUTPUT_DIR, "model_quantized.onnx")
    print("Quantizing model to INT8...")
    try:
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QInt8,
        )
        print(f"✓ Quantized model saved to {quantized_path}")
        
        # Check file sizes
        orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"  Original: {orig_size:.2f} MB")
        print(f"  Quantized: {quant_size:.2f} MB")
        print(f"  Compression: {100 * (1 - quant_size/orig_size):.1f}%")
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        print(f"  Will use Float32 model at {onnx_path}")

if __name__ == "__main__":
    export_model()
