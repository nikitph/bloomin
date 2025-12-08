
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def embed_wiki():
    data_dir = "data"
    output_dir = "faiss-sphere-rs/data"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, "wiki.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Reading wiki.txt...")
    max_docs = 10000
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 50: # filter short lines
                docs.append(line)
                if len(docs) >= max_docs:
                    break
    
    if len(docs) == 0:
        print("No documents found!")
        return

    print(f"Embedding {len(docs)} documents...")
    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings = embeddings.astype('float32') # Important for Rust
    
    print(f"Saving documents.npy: {embeddings.shape}")
    np.save(os.path.join(output_dir, "documents.npy"), embeddings)
    
    # Generate queries (use first 100 docs as queries for simplicity, or some other strategy)
    # Actually let's just split: last 100 as queries
    if len(docs) > 100:
        queries = embeddings[-100:]
        docs_vecs = embeddings[:-100]
    else:
        queries = embeddings
        docs_vecs = embeddings
        
    print(f"Saving queries.npy: {queries.shape}")
    np.save(os.path.join(output_dir, "queries.npy"), queries)
    
    # Save the doc vectors separately if we split
    print(f"Saving final documents.npy: {docs_vecs.shape}")
    np.save(os.path.join(output_dir, "documents.npy"), docs_vecs)

if __name__ == "__main__":
    embed_wiki()
