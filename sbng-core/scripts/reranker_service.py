#!/usr/bin/env python3
"""
GPU-accelerated re-ranker microservice using PyTorch MPS (Apple Silicon).
"""
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
import uvicorn

# Configuration
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-2-v2"
PORT = 8001

# Check for MPS (Apple Silicon GPU) availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✓ Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print(f"⚠ Using CPU (no GPU available)")

# Load model and tokenizer
print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.to(device)
model.eval()
print(f"✓ Model loaded on {device}")

app = FastAPI(title="GPU Re-ranker Service")

class RerankRequest(BaseModel):
    query: str
    candidates: List[Tuple[str, str]]  # [(doc_id, doc_text), ...]

class RerankResponse(BaseModel):
    results: List[Tuple[str, float]]  # [(doc_id, score), ...] sorted by score desc

@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    """Re-rank candidates using cross-encoder model on GPU."""
    if not request.candidates:
        return RerankResponse(results=[])
    
    # Prepare inputs
    query = request.query
    doc_ids = [doc_id for doc_id, _ in request.candidates]
    doc_texts = [doc_text for _, doc_text in request.candidates]
    
    # Tokenize all pairs
    inputs = tokenizer(
        [query] * len(doc_texts),
        doc_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)  # Shape: [batch_size]
        scores = logits.cpu().tolist()
    
    # Combine and sort
    results = list(zip(doc_ids, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    return RerankResponse(results=results)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

if __name__ == "__main__":
    print(f"\nStarting re-ranker service on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
