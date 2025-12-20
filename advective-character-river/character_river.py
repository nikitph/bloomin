
import numpy as np
import time
from collections import defaultdict
import requests
import os

class CharacterRiver:
    """Advective Bloom Filter for character prediction (2D Version)"""
    
    def __init__(self, side_len=16):
        """
        Map characters to a 2D side_len x side_len grid.
        For 256 chars, 16x16 is perfect.
        """
        self.side = side_len
        self.grid_size = side_len * side_len
        
        # Flow field: velocity vectors (vy, vx) at each character position
        self.flow = np.zeros((self.grid_size, 2), dtype=np.float32)
        
        # Metadata
        self.transition_counts = np.zeros(self.grid_size, dtype=np.int32)
        
    def _char_to_pos(self, char):
        """Map character to (row, col)"""
        idx = ord(char) % self.grid_size
        return idx // self.side, idx % self.side
    
    def _pos_to_char(self, r, c):
        """Map (row, col) back to character"""
        # Wrap coords
        r = int(round(r)) % self.side
        c = int(round(c)) % self.side
        idx = r * self.side + c
        return chr(idx)
    
    def train(self, text, learning_rate=0.5):
        """
        Learn flow field from text
        """
        # Filter mostly to printable to avoid noise, but keep newlines
        valid_chars = set([chr(i) for i in range(32, 127)] + ['\n', '\t'])
        
        # Pre-calculate positions
        indices = [ord(c) % self.grid_size for c in text]
        rows = np.array([i // self.side for i in indices])
        cols = np.array([i % self.side for i in indices])
        
        for i in range(len(text) - 1):
            curr_r, curr_c = rows[i], cols[i]
            next_r, next_c = rows[i+1], cols[i+1]
            idx = curr_r * self.side + curr_c
            
            # Velocity with toroidal wrapping logic
            dr = next_r - curr_r
            dc = next_c - curr_c
            
            # Shortest path on torus
            if dr > self.side // 2: dr -= self.side
            elif dr < -self.side // 2: dr += self.side
            
            if dc > self.side // 2: dc -= self.side
            elif dc < -self.side // 2: dc += self.side
            
            # Update flow (Test: High alpha to mimic "Last Seen" instead of "Mean")
            # alpha = learning_rate / (1 + cnt * 0.05) 
            alpha = learning_rate # Constant alpha keeps it plastic (LRU)
            
            self.flow[idx, 0] += alpha * (dr - self.flow[idx, 0])
            self.flow[idx, 1] += alpha * (dc - self.flow[idx, 1])
            
            self.transition_counts[idx] += 1
            
    def predict_next(self, char, num_steps=1):
        idx = ord(char) % self.grid_size
        r, c = idx // self.side, idx % self.side
        
        # Get velocity
        vr = self.flow[idx, 0]
        vc = self.flow[idx, 1]
        
        # Advect
        pred_r = r + vr * num_steps
        pred_c = c + vc * num_steps
        
        return self._pos_to_char(pred_r, pred_c)
    
    def predict_sequence(self, seed, length=100):
        result = seed
        current = seed[-1]
        for _ in range(length):
            next_char = self.predict_next(current)
            result += next_char
            current = next_char
        return result


class UnigramBaseline:
    """1-gram predictor (Most Frequent Next Character)"""
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
    
    def train(self, text):
        for i in range(len(text) - 1):
            curr = text[i]
            nxt = text[i+1]
            self.counts[curr][nxt] += 1
            
    def predict_next(self, context):
        char = context[-1]
        if char not in self.counts: return ' '
        return max(self.counts[char].items(), key=lambda x: x[1])[0]


class NGramBaseline:
    """Standard n-gram predictor for comparison"""
    
    def __init__(self, n=3):
        self.n = n
        self.transitions = defaultdict(lambda: defaultdict(int))
    
    def train(self, text):
        for i in range(len(text) - self.n):
            context = text[i:i+self.n]
            next_char = text[i+self.n]
            self.transitions[context][next_char] += 1
    
    def predict_next(self, context):
        if len(context) < self.n:
            return ' '
        
        context = context[-self.n:]
        
        if context not in self.transitions:
            return ' '
        
        candidates = self.transitions[context]
        return max(candidates.items(), key=lambda x: x[1])[0]


def download_shakespeare():
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    filename = "shakespeare.txt"
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            r = requests.get(url)
            r.encoding = 'utf-8' # Ensure correct encoding
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(r.text)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download: {e}")
            # Create a dummy file if download fails
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("To be or not to be, that is the question. " * 1000)
    else:
        print(f"{filename} already exists.")

# ============ THE EXPERIMENT ============

def run_character_prediction_experiment():
    """
    Compare advective vs n-gram on character prediction
    """
    download_shakespeare()
    
    # Load text (use any public domain book)
    print("Loading training data...")
    try:
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            # Clean text a bit, remove BOM if present
            if text.startswith('\ufeff'):
                text = text[1:]
            # Limit to 100k or full text if smaller
            text = text[:100000]
    except FileNotFoundError:
        print("Error: shakespeare.txt not found.")
        return
    
    # Split train/test
    split = int(0.8 * len(text))
    train_text = text[:split]
    test_text = text[split:]
    
    print(f"Training on {len(train_text)} characters...")
    print(f"Testing on {len(test_text)} characters...")
    
    # ===== Train Advective =====
    print("\n[1/2] Training Advective Character River...")
    river = CharacterRiver(side_len=16)
    
    start = time.time()
    river.train(train_text, learning_rate=0.2)
    train_time_river = time.time() - start
    
    print(f"  Trained in {train_time_river:.3f}s")
    print(f"  Throughput: {len(train_text)/train_time_river:.0f} chars/sec")
    
    # ===== Train N-Gram =====
    print("\n[2/2] Training 3-gram baseline...")
    ngram = NGramBaseline(n=3)
    
    start = time.time()
    ngram.train(train_text)
    train_time_ngram = time.time() - start
    
    print(f"  Trained in {train_time_ngram:.3f}s")
    
    # ===== Train Unigram =====
    print("\n[3/3] Training 1-gram baseline...")
    unigram = UnigramBaseline()
    unigram.train(train_text)
    
    # ===== Evaluate =====
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    def evaluate_predictor(predictor, name, context_length=1):
        correct = 0
        total = 0
        
        prediction_times = []
        
        # Test on 1000 random positions
        test_samples = 1000
        # Ensure we have enough context
        valid_start = context_length
        if len(test_text) <= valid_start + 1:
            print("Not enough test data.")
            return 0, 0
            
        for _ in range(test_samples):
            pos = np.random.randint(valid_start, len(test_text) - 1)
            
            if context_length == 1:
                context = test_text[pos]
            else:
                context = test_text[pos-context_length:pos]
            
            actual_next = test_text[pos]
            
            # Time the prediction
            start = time.perf_counter()
            predicted = predictor.predict_next(context)
            elapsed = time.perf_counter() - start
            
            prediction_times.append(elapsed)
            
            if predicted == actual_next:
                correct += 1
            total += 1
        
        accuracy = correct / total
        avg_latency_us = np.mean(prediction_times) * 1e6
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Latency: {avg_latency_us:.3f} μs")
        print(f"  Throughput: {1e6/avg_latency_us:.0f} predictions/sec")
        
        return accuracy, avg_latency_us
    
    river_acc, river_latency = evaluate_predictor(river, "Advective River", context_length=1)
    ngram_acc, ngram_latency = evaluate_predictor(ngram, "3-gram Baseline", context_length=3)
    uni_acc, uni_latency = evaluate_predictor(unigram, "1-gram Baseline", context_length=1)
    
    # ===== Results Summary =====
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    speedup = ngram_latency / (river_latency + 1e-9)
    accuracy_ratio = river_acc / (ngram_acc + 1e-9)
    
    print(f"\nSpeed: {speedup:.1f}x faster")
    print(f"Accuracy: {accuracy_ratio*100:.1f}% of baseline")
    
    # Memory comparison
    river_memory = river.flow.nbytes / 1024  # KB
    
    # Rough N-gram size estimate: keys (chars * n) + values (dict overhead)
    # A cleaner way to estimate: number of entries
    ngram_entries = sum(len(v) for v in ngram.transitions.values())
    # Assuming minimal python dict overhead per entry (~100 bytes is conservative)
    ngram_memory = ngram_entries * 0.1 # Estimate in KB
    if hasattr(ngram, 'transitions'):
         import sys
         # This is still rough as it doesn't count nested structure fully
         ngram_memory = sys.getsizeof(ngram.transitions) / 1024 
         for k, v in ngram.transitions.items():
            ngram_memory += sys.getsizeof(v) / 1024
    
    print(f"\nMemory:")
    print(f"  Advective: {river_memory:.2f} KB")
    print(f"  N-gram: {ngram_memory:.2f} KB")
    print(f"  Ratio: {river_memory/(ngram_memory+1e-9):.2f}x")
    
    # ===== Qualitative Demo =====
    print("\n" + "="*60)
    print("QUALITATIVE DEMO: Text Generation")
    print("="*60)
    
    seed = "To be or not to be"
    
    print(f"\nSeed: '{seed}'")
    print(f"\nAdvective continuation:")
    print(river.predict_sequence(seed, length=100))
    
    # Success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    
    if speedup > 5:
        print(f"✓ Speed: {speedup:.1f}x faster (need >5x)")
    else:
        print(f"✗ Speed: Only {speedup:.1f}x faster")
    
    if accuracy_ratio > 0.5:
        print(f"✓ Accuracy: {accuracy_ratio*100:.1f}% of baseline (need >50%)")
    else:
        print(f"✗ Accuracy: Only {accuracy_ratio*100:.1f}% of baseline")
    
    return {
        'speedup': speedup,
        'accuracy_ratio': accuracy_ratio,
        'memory': river_memory
    }


if __name__ == "__main__":
    results = run_character_prediction_experiment()
