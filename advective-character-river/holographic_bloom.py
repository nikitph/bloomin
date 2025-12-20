
import numpy as np
import time
from collections import defaultdict
import os

class HolographicBloomFilter:
    """
    Primitive 50: The Holographic Bloom Filter.
    Uses Hyperdimensional Computing (HDC) to store transitions in superposition
    rather than averaging them into noise.
    """
    def __init__(self, vocab_size=65536, dim=2048, table_size=65536):
        self.dim = dim
        self.vocab_size = vocab_size
        self.table_size = table_size
        
        # 1. Dictionary: Random Hypervectors (-1, +1)
        # Using int8 to save memory, but math operations will need promotion
        print(f"Initializing Holographic Bloom Filter (Dim={dim})...")
        self.vocab = np.random.choice([-1, 1], size=(vocab_size, dim)).astype(np.int8)
        
        # 2. The Table: Accumulators
        self.table = np.zeros((table_size, dim), dtype=np.int16)
        
        # Optimization: Track seen characters to speed up prediction
        self.seen_indices = set()
        
    def train(self, text):
        """
        Train on text sequence.
        O(N) complexity. 
        """
        print(f"Training on {len(text)} characters...")
        
        for i in range(len(text) - 1):
            char_in = ord(text[i])
            char_out = ord(text[i+1])
            
            # Hash Input to Table Index
            h = (char_in * 0x811c9dc5) % self.table_size
            
            # Superimpose Output Vector
            self.table[h] += self.vocab[char_out]
            
            # Track seen char
            self.seen_indices.add(char_out)
            
    def predict_next(self, char_in, top_k=3):
        """
        Predict next characters by probing the superposition.
        """
        # Hash Input
        h = (ord(char_in) * 0x811c9dc5) % self.table_size
        
        # Retrieve the bundle
        memory_vec = self.table[h]
        
        # Optimization: Only check against seen characters
        active_idxs = list(self.seen_indices)
        if not active_idxs:
            return []
            
        active_vocab = self.vocab[active_idxs]
        
        # Query: "Who is in the bundle?"
        # Dot product with active vocabulary vectors
        scores = np.dot(active_vocab.astype(np.int32), memory_vec.astype(np.int32))
        
        # Get indices relative to active_vocab
        k = min(top_k, len(scores))
        best_local_indices = np.argpartition(scores, -k)[-k:]
        
        # Sort these top-k
        best_local_indices = best_local_indices[np.argsort(scores[best_local_indices])[::-1]]
        
        # Map back to global indices (char/ord)
        return [chr(active_idxs[i]) for i in best_local_indices]

class UnigramBaseline:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
    
    def train(self, text):
        for i in range(len(text) - 1):
            self.counts[text[i]][text[i+1]] += 1
            
    def predict(self, char):
        if char not in self.counts: return []
        return [k for k, v in sorted(self.counts[char].items(), key=lambda x: x[1], reverse=True)[:3]]

def run_holographic_experiment():
    print("="*60)
    print("HOLOGRAPHIC BLOOM FILTER EXPERIMENTAL RUN")
    print("="*60)
    
    # Load Data
    try:
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()[:100000] # 100k chars
    except FileNotFoundError:
        text = "To be or not to be " * 5000
    
    split = int(0.8 * len(text))
    train_text = text[:split]
    test_text = text[split:]
    
    # 1. Train Holographic Model
    # Dim=2048 provides robustness
    holo = HolographicBloomFilter(dim=2048) 
    
    start = time.time()
    holo.train(train_text)
    holo_time = time.time() - start
    print(f"Holographic Train Time: {holo_time:.3f}s")
    
    # 2. Train Baseline
    baseline = UnigramBaseline()
    start = time.time()
    baseline.train(train_text)
    base_time = time.time() - start
    print(f"Baseline Train Time:    {base_time:.3f}s")
    
    # 3. Evaluate
    print("\nEvaluating...")
    
    holo_top1 = 0
    holo_top3 = 0
    base_top1 = 0
    base_top3 = 0
    total = 0
    
    holo_latencies = []
    base_latencies = []
    
    # Test 1000 samples
    indices = np.random.randint(0, len(test_text)-1, 1000)
    
    for idx in indices:
        char_in = test_text[idx]
        actual_next = test_text[idx+1]
        
        # Holographic Predict
        t0 = time.perf_counter()
        holo_preds = holo.predict_next(char_in, top_k=3)
        t1 = time.perf_counter()
        holo_latencies.append(t1-t0)
        
        # Baseline Predict
        t0 = time.perf_counter()
        base_preds = baseline.predict(char_in)
        t1 = time.perf_counter()
        base_latencies.append(t1-t0)
        
        # Score Holo
        if holo_preds:
            if holo_preds[0] == actual_next: holo_top1 += 1
            if actual_next in holo_preds: holo_top3 += 1
            
        # Score Base
        if base_preds:
            if base_preds[0] == actual_next: base_top1 += 1
            if actual_next in base_preds: base_top3 += 1
            
        total += 1
        
    # Results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"Holographic (2048D):")
    print(f"  Top-1 Accuracy: {100*holo_top1/total:.2f}%")
    print(f"  Top-3 Accuracy: {100*holo_top3/total:.2f}%")
    print(f"  Avg Latency:    {np.mean(holo_latencies)*1e6:.2f} μs")
    print(f"  Memory (Est):   {holo.table.nbytes/1024:.0f} KB (Table) + {holo.vocab.nbytes/1024:.0f} KB (Vocab)")
    
    print(f"\nUnigram Baseline:")
    print(f"  Top-1 Accuracy: {100*base_top1/total:.2f}%")
    print(f"  Top-3 Accuracy: {100*base_top3/total:.2f}%")
    print(f"  Avg Latency:    {np.mean(base_latencies)*1e6:.2f} μs")
    
    # Analysis
    print("\nAnalysis:")
    acc_ratio = holo_top1 / (base_top1 + 1e-9)
    print(f"  Accuracy Ratio: {acc_ratio*100:.1f}%")
    
    if acc_ratio > 0.9:
        print("  ✓ MULTIMODALITY SOLVED: Superposition preserves distinct futures!")
    else:
        print("  ✗ Issues remain (probably hash collisions or noise floor")

    # Qualitative Demo
    print("\nDemo Generation:")
    seed = "T"
    curr = seed
    out = seed
    for _ in range(50):
        preds = holo.predict_next(curr)
        if preds:
            curr = preds[0] # Greedy
            out += curr
        else:
            break
    print(f"Holo: '{out}'")

if __name__ == "__main__":
    run_holographic_experiment()
