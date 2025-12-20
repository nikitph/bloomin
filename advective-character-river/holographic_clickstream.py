
import numpy as np
import time
from collections import defaultdict
import os
import hashlib

class HolographicClickstream:
    """
    Holographic Memory for Web Clicks.
    """
    def __init__(self, dim=2048, table_size=65536):
        self.dim = dim
        self.table_size = table_size
        
        # 1. Dictionary: Lazy Hypervector Generation
        # Maps page_name -> index in self.vocab_matrix
        self.page_to_idx = {}
        self.idx_to_page = []
        
        # We start with empty vocab matrix and grow it
        # Or better: pre-allocate a chunk and grow? 
        # For simplicity in this script, we'll list grow and stack at query time? 
        # No, slow. Let's precise pre-scan or dynamic append.
        self.vocab_matrix = np.zeros((0, dim), dtype=np.int8)
        
        # 2. The Table: Accumulators
        # Using int32 because click weights can be large
        self.table = np.zeros((table_size, dim), dtype=np.int32)
        
    def get_or_create_vector_idx(self, page):
        if page in self.page_to_idx:
            return self.page_to_idx[page]
        
        # Generate stable random vector
        # deterministic hash -> seed
        hash_digest = hashlib.md5(page.encode('utf-8')).digest()
        seed = int.from_bytes(hash_digest[:4], 'big')
        rng = np.random.RandomState(seed)
        vec = rng.choice([-1, 1], size=self.dim).astype(np.int8)
        
        # Add to vocab
        idx = len(self.idx_to_page)
        self.page_to_idx[page] = idx
        self.idx_to_page.append(page)
        
        # Append to matrix (inefficient for growing, but ok for batch loading)
        if len(self.vocab_matrix) == 0:
            self.vocab_matrix = vec.reshape(1, -1)
        else:
            self.vocab_matrix = np.vstack([self.vocab_matrix, vec])
            
        return idx

    def train(self, transitions):
        """
        transitions: list of (source, dest, count)
        """
        print(f"Training on {len(transitions)} transitions...")
        
        # Pre-scan to build vocab matrix efficiently?
        # That would save vstack costs.
        unique_pages = set()
        for s, d, c in transitions:
            unique_pages.add(s)
            unique_pages.add(d)
        
        print(f"  Unique pages (Vocabulary): {len(unique_pages)}")
        
        # Build Vocab Matrix in one shot
        sorted_pages = sorted(list(unique_pages))
        self.idx_to_page = sorted_pages
        self.page_to_idx = {p: i for i, p in enumerate(sorted_pages)}
        
        # Generate vectors
        # Use a single large RNG generation for speed
        rng = np.random.RandomState(42)
        self.vocab_matrix = rng.choice([-1, 1], size=(len(sorted_pages), self.dim)).astype(np.int8)
        
        print("  Vocabulary Matrix Built.")
        
        
        # Encode Transitions
        for src, dest, count in transitions:
            # Hash Source to Table Index
            # FNV-1a style
            h = (hash(src) * 0x811c9dc5) % self.table_size
            
            # Get Dest Vector index
            dst_idx = self.page_to_idx[dest]
            
            # Superimpose weighted vector
            self.table[h] += self.vocab_matrix[dst_idx].astype(np.int32) * int(count)
        
        # Pre-cast for fast dot product in prediction
        self.vocab_matrix = self.vocab_matrix.astype(np.float32)

    def predict_next(self, src, top_k=5):
        # Hash Source
        h = (hash(src) * 0x811c9dc5) % self.table_size
        
        # Retrieve Bundle
        bundle = self.table[h].astype(np.float32)
        
        # Query against entire vocabulary
        # Scores = Vocab @ Bundle
        scores = np.dot(self.vocab_matrix, bundle)
        
        # Top K
        if len(scores) < top_k:
            best_indices = np.argsort(scores)[::-1]
        else:
            best_indices = np.argpartition(scores, -top_k)[-top_k:]
            best_indices = best_indices[np.argsort(scores[best_indices])[::-1]]
            
        return [self.idx_to_page[i] for i in best_indices]

class MarkovBaseline:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
    
    def train(self, transitions):
        for src, dst, count in transitions:
            self.transitions[src][dst] += count
    
    def predict_next(self, src):
        if src not in self.transitions: return None
        return max(self.transitions[src].items(), key=lambda x: x[1])[0]
        
    def predict_top_k(self, src, k=5):
        if src not in self.transitions: return []
        return [x[0] for x in sorted(self.transitions[src].items(), key=lambda x: x[1], reverse=True)[:k]]

def run_clickstream_experiment(data_path='clickstream_sample.tsv'):
    print("="*60)
    print("HOLOGRAPHIC CLICKSTREAM EXPERIMENT")
    print("="*60)
    
    # Load Data (Reuse usage of existing file)
    transitions = []
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Using dummy data.")
        # Dummy
        return
        
    with open(data_path, 'r', encoding='utf-8') as f:
        # Check standard wiki format
        for line in f:
            parts = line.strip().split('\t')
            # Format: prev curr type n
            if len(parts) >= 4 and parts[2] == 'link':
                try:
                    transitions.append((parts[0], parts[1], int(parts[3])))
                except: pass
            # Fallback format: prev curr count
            elif len(parts) >= 3 and parts[2].isdigit():
                 transitions.append((parts[0], parts[1], int(parts[2])))
                 
    print(f"Loaded {len(transitions)} transitions")
    
    # Shuffle
    np.random.shuffle(transitions)
    
    # Split
    split = int(0.8 * len(transitions))
    train_data = transitions[:split]
    test_data = transitions[split:]
    
    # 1. Train Holograph
    # Using 4096 dim for robustness with larger vocabulary (60k pages)
    holo = HolographicClickstream(dim=4096, table_size=131072) 
    
    start = time.time()
    holo.train(train_data)
    holo_time = time.time() - start
    print(f"Holographic Train Time: {holo_time:.3f}s")
    
    # 2. Train Markov
    markov = MarkovBaseline()
    start = time.time()
    markov.train(train_data)
    base_time = time.time() - start
    print(f"Markov Train Time:      {base_time:.3f}s")
    
    # 3. Evaluate
    print("\nEvaluating...")
    
    holo_top1 = 0
    holo_top5 = 0
    base_top1 = 0
    base_top5 = 0
    total = 0
    
    holo_latencies = []
    
    # Limit test samples for speed (Matrix mult is heavy)
    test_limit = 1000
    
    for src, dst, _ in test_data[:test_limit]:
        # Holograph
        t0 = time.perf_counter()
        h_preds = holo.predict_next(src, top_k=5)
        t1 = time.perf_counter()
        holo_latencies.append(t1-t0)
        
        # Markov
        m_preds = markov.predict_top_k(src, k=5)
        
        # Score Holo
        if h_preds:
            if h_preds[0] == dst: holo_top1 += 1
            if dst in h_preds: holo_top5 += 1
            
        # Score Markov
        if m_preds:
            if m_preds[0] == dst: base_top1 += 1
            if dst in m_preds: base_top5 += 1
            
        total += 1
        
    print("\nRESULTS SUMMARY")
    print(f"Holographic (4096D):")
    print(f"  Top-1 Accuracy: {100*holo_top1/total:.2f}%")
    print(f"  Top-5 Accuracy: {100*holo_top5/total:.2f}%")
    print(f"  Avg Latency:    {np.mean(holo_latencies)*1e6:.2f} μs")
    
    print(f"Markov Baseline:")
    print(f"  Top-1 Accuracy: {100*base_top1/total:.2f}%")
    print(f"  Top-5 Accuracy: {100*base_top5/total:.2f}%")
    
    # Memory check
    holo_mem = (holo.table.nbytes + holo.vocab_matrix.nbytes) / 1024 / 1024
    import sys
    markov_mem = sys.getsizeof(markov.transitions) / 1024 / 1024 # rough
    print(f"\nMemory (MB): Holo={holo_mem:.1f} vs Markov=~{markov_mem:.1f}")

if __name__ == "__main__":
    run_clickstream_experiment()
