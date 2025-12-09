"""
Wikipedia Dataset Preparation for SemantixBit Benchmark

Downloads Wikipedia dataset, chunks into passages, and generates embeddings
using sentence-transformers (MiniLM model).
"""

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import os

# Configuration
DATASET_SIZE = 100_000  # Number of Wikipedia articles
CHUNK_SIZE = 200  # Words per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "data/wikipedia"

def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks

def prepare_wikipedia_data():
    """Download and prepare Wikipedia dataset"""
    print("Loading Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process articles
    passages = []
    embeddings = []
    metadata = []
    
    print(f"Processing {DATASET_SIZE} Wikipedia articles...")
    for i, article in enumerate(tqdm(dataset, total=DATASET_SIZE)):
        if i >= DATASET_SIZE:
            break
        
        title = article['title']
        text = article['text']
        
        # Skip very short articles
        if len(text) < 100:
            continue
        
        # Chunk article into passages
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for j, chunk in enumerate(chunks):
            passage_id = f"{title}__chunk_{j}"
            passages.append(chunk)
            metadata.append({
                'id': passage_id,
                'title': title,
                'chunk_idx': j,
                'article_idx': i
            })
    
    print(f"Generated {len(passages)} passages from {DATASET_SIZE} articles")
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    batch_size = 32
    for i in tqdm(range(0, len(passages), batch_size)):
        batch = passages[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Normalize embeddings (important for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save data
    print("Saving data...")
    np.save(f"{OUTPUT_DIR}/embeddings.npy", embeddings)
    
    with open(f"{OUTPUT_DIR}/passages.json", 'w') as f:
        json.dump(passages, f)
    
    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    # Save dataset info
    info = {
        'num_passages': len(passages),
        'num_articles': DATASET_SIZE,
        'embedding_dim': embedding_dim,
        'model_name': MODEL_NAME,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP
    }
    
    with open(f"{OUTPUT_DIR}/info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Dataset saved to {OUTPUT_DIR}")
    print(f"Total passages: {len(passages)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Memory usage: {embeddings.nbytes / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    prepare_wikipedia_data()
