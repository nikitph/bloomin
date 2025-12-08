"""
Generate real Wikipedia embeddings using BERT

This script:
1. Downloads Wikipedia articles using datasets library
2. Encodes them with BERT-base model
3. Saves embeddings for benchmarking
"""

import numpy as np
import os
from tqdm import tqdm

def generate_wikipedia_bert_embeddings(
    n_documents: int = 10000,
    n_queries: int = 1000,
    output_dir: str = 'data'
):
    """
    Generate Wikipedia embeddings using BERT
    
    Args:
        n_documents: Number of Wikipedia articles to encode
        n_queries: Number of query embeddings to generate
        output_dir: Directory to save embeddings
    """
    print("="*80)
    print("GENERATING WIKIPEDIA + BERT EMBEDDINGS")
    print("="*80)
    
    # Install required packages if needed
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("\nInstalling required packages...")
        import subprocess
        subprocess.check_call([
            'pip3', 'install', 
            'datasets', 'transformers', 'torch'
        ])
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModel
        import torch
    
    # Load BERT model
    print("\nLoading BERT model...")
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Use GPU if available
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load Wikipedia dataset with STREAMING (only downloads what we need!)
    print(f"\nLoading Wikipedia dataset ({n_documents} articles)...")
    print("Using streaming mode to avoid downloading entire dataset...")
    
    dataset = load_dataset(
        'wikimedia/wikipedia',
        '20231101.en',
        split='train',
        streaming=True  # KEY: Only download what we iterate over!
    )
    
    # Take only what we need
    print(f"Collecting {n_documents + n_queries} articles...")
    articles = []
    for i, article in enumerate(dataset):
        if i >= n_documents + n_queries:
            break
        articles.append(article)
        if (i + 1) % 1000 == 0:
            print(f"  Collected {i + 1} articles...")
    
    print(f"Loaded {len(articles)} articles")
    
    # Encode function
    def encode_text(text, max_length=512):
        """Encode text to BERT embedding"""
        # Tokenize
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    # Encode documents
    print("\nEncoding documents...")
    documents = []
    for i in tqdm(range(n_documents)):
        text = articles[i]['text'][:1000]  # Use first 1000 chars
        if not text.strip():
            text = articles[i]['title']  # Fallback to title
        
        embedding = encode_text(text)
        documents.append(embedding)
    
    documents = np.array(documents, dtype='float32')
    
    # Encode queries (use remaining articles)
    print("\nEncoding queries...")
    queries = []
    for i in tqdm(range(n_documents, n_documents + n_queries)):
        text = articles[i]['text'][:1000]
        if not text.strip():
            text = articles[i]['title']
        
        embedding = encode_text(text)
        queries.append(embedding)
    
    queries = np.array(queries, dtype='float32')
    
    # Normalize
    print("\nNormalizing embeddings...")
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    
    doc_path = os.path.join(output_dir, 'wikipedia_documents.npy')
    query_path = os.path.join(output_dir, 'wikipedia_queries.npy')
    
    print(f"\nSaving embeddings...")
    np.save(doc_path, documents)
    np.save(query_path, queries)
    
    print(f"\n✅ Saved:")
    print(f"  Documents: {doc_path} ({documents.shape})")
    print(f"  Queries: {query_path} ({queries.shape})")
    print(f"  Dimension: {documents.shape[1]}D")
    
    return documents, queries


if __name__ == '__main__':
    # Generate embeddings
    docs, queries = generate_wikipedia_bert_embeddings(
        n_documents=100000,
        n_queries=1000
    )
    
    print("\n" + "="*80)
    print("EMBEDDINGS READY FOR BENCHMARKING!")
    print("="*80)
    print("\nRun: cd experiments && python3 wikipedia_benchmark_real.py")
