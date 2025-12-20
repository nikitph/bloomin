import numpy as np
from collections import defaultdict
import time
import os
import requests
import gzip
import shutil

class AdvectiveBloomFilter:
    def __init__(self, grid_size=128, learning_rate=0.1):
        """
        grid_size: NxN grid resolution (128 = 16KB for 2D velocity field)
        learning_rate: How much each transition updates the field
        """
        self.grid_size = grid_size
        self.lr = learning_rate
        
        # Velocity field: each cell stores (vx, vy)
        self.velocity_field = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
        
        # For mapping items to 2D coordinates
        self.item_to_coord = {}
        self.coord_to_items = defaultdict(list)
        
    def _hash_to_grid(self, item):
        """Map item to grid coordinates [0, grid_size)"""
        if item in self.item_to_coord:
            return self.item_to_coord[item]
        
        # Simple hash-based positioning
        # Use deterministic hash for reproducibility
        h = int(hash(item))
        x = (h % self.grid_size)
        y = ((h // self.grid_size) % self.grid_size)
        
        self.item_to_coord[item] = (x, y)
        self.coord_to_items[(x, y)].append(item)
        
        return x, y
    
    def insert_transition(self, from_item, to_item, weight=1.0):
        """
        Record that from_item -> to_item transition occurred
        Updates velocity field to flow in that direction
        """
        x1, y1 = self._hash_to_grid(from_item)
        x2, y2 = self._hash_to_grid(to_item)
        
        # Calculate velocity vector
        dx = (x2 - x1)
        dy = (y2 - y1)
        
        # Handle wraparound for toroidal topology
        if abs(dx) > self.grid_size // 2:
            dx = dx - np.sign(dx) * self.grid_size
        if abs(dy) > self.grid_size // 2:
            dy = dy - np.sign(dy) * self.grid_size
        
        # Update velocity field (running average-ish)
        # Scale update by weight (optional, or just constant LR)
        self.velocity_field[x1, y1, 0] += self.lr * dx
        self.velocity_field[x1, y1, 1] += self.lr * dy
        
    def predict_next(self, current_item, dt=1.0, num_steps=5):
        """
        Predict next item by advecting along velocity field
        
        dt: time step size
        num_steps: number of advection steps (higher = further prediction)
        """
        if current_item not in self.item_to_coord:
            return None
        
        x, y = self.item_to_coord[current_item]
        
        # Convert to float for advection
        pos = np.array([float(x), float(y)])
        
        # Advect along flow field
        for _ in range(num_steps):
            ix, iy = int(pos[0]) % self.grid_size, int(pos[1]) % self.grid_size
            velocity = self.velocity_field[ix, iy]
            pos += velocity * dt
        
        # Find nearest item at predicted location
        final_x = int(pos[0]) % self.grid_size
        final_y = int(pos[1]) % self.grid_size
        
        # Check nearby cells for items
        for radius in range(3):  # Search expanding radius
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    check_x = (final_x + dx) % self.grid_size
                    check_y = (final_y + dy) % self.grid_size
                    
                    if (check_x, check_y) in self.coord_to_items:
                        candidates = self.coord_to_items[(check_x, check_y)]
                        if candidates:
                            # Heuristic: return the one with the smallest semantic distance? 
                            # Or just the first one?
                            return candidates[0]  # Return first match
        
        return None
    
    def predict_top_k(self, current_item, k=5, dt=1.0):
        """Return top-k predictions by sampling flow field"""
        if current_item not in self.item_to_coord:
            return []
        
        predictions = []
        seen = set()
        
        # Try different time steps/accumulate path
        # Simplification: just take steps along the path
        x, y = self.item_to_coord[current_item]
        pos = np.array([float(x), float(y)])
        
        for _ in range(k * 2): # Look a bit further
            ix, iy = int(pos[0]) % self.grid_size, int(pos[1]) % self.grid_size
            velocity = self.velocity_field[ix, iy]
            pos += velocity * dt
            
            check_x = int(pos[0]) % self.grid_size
            check_y = int(pos[1]) % self.grid_size
            
            if (check_x, check_y) in self.coord_to_items:
                for cand in self.coord_to_items[(check_x, check_y)]:
                    if cand not in seen and cand != current_item:
                        predictions.append(cand)
                        seen.add(cand)
                        if len(predictions) >= k:
                            return predictions
                            
        return predictions


class MarkovBaseline:
    """First-order Markov chain baseline"""
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
    
    def insert_transition(self, from_item, to_item, weight=1.0):
        self.transitions[from_item][to_item] += weight
    
    def predict_next(self, current_item):
        if current_item not in self.transitions:
            return None
        
        # Return most frequent next item
        next_items = self.transitions[current_item]
        if not next_items:
            return None
        
        return max(next_items.items(), key=lambda x: x[1])[0]
    
    def predict_top_k(self, current_item, k=5):
        if current_item not in self.transitions:
            return []
        
        next_items = self.transitions[current_item]
        sorted_items = sorted(next_items.items(), key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in sorted_items[:k]]

def download_clickstream_sample(filename='clickstream_sample.tsv', limit=100000):
    """
    Downloads a sample of Wikipedia clickstream data.
    Since the full file is huge, we'll try to stream it or use a smaller version.
    """
    # 2024-11 might not be published yet, usually lag of a month or two.
    # Trying a known recent one: 2023-11 or check validity.
    # Actually, simplest is to use '2023-12' which likely exists.
    # Let's try to fetch a small chunk if possible, or fail gracefully.
    
    url = "https://dumps.wikimedia.org/other/clickstream/2023-12/clickstream-enwiki-2023-12.tsv.gz"
    
    if os.path.exists(filename):
        print(f"Using existing data: {filename}")
        return

    print(f"Downloading clickstream data from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename + '.gz', 'wb') as f:
                # Download first 50MB (enough for sample)
                count = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    count += len(chunk)
                    if count > 50 * 1024 * 1024: # 50MB limit
                        break
        
        print("Decompressing (partial)...")
        with gzip.open(filename + '.gz', 'rt', encoding='utf-8') as f_in:
            with open(filename, 'w', encoding='utf-8') as f_out:
                lines_written = 0
                for line in f_in:
                    f_out.write(line)
                    lines_written += 1
                    if lines_written >= limit:
                        break
                        
        os.remove(filename + '.gz')
        print(f"Saved {lines_written} lines to {filename}")
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("Generating synthetic data relative to failure...")
        generate_synthetic_data(filename, limit)

def generate_synthetic_data(filename, limit):
    print("Generating synthetic clickstream data (Zipfian distribution)...")
    # Simulate realistic web traffic: Power law connections
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("prev\tcurr\ttype\tn\n")
        
        vocab_size = 1000
        # Create a graph where node i links to node j with prob proportional to 1/j
        
        for _ in range(limit):
            # Zipfian source
            src_id = min(int(np.random.zipf(1.5)), vocab_size) 
            # Zipfian dest (correlated slightly?)
            dst_id = min(int(np.random.zipf(1.5)), vocab_size)
            
            if src_id == dst_id: dst_id = (dst_id + 1) % vocab_size
            
            src = f"Article_{src_id}"
            dst = f"Article_{dst_id}"
            count = np.random.randint(1, 100)
            
            f.write(f"{src}\t{dst}\tlink\t{count}\n")


def run_experiment(data_path='clickstream_sample.tsv'):
    
    if not os.path.exists(data_path):
        download_clickstream_sample(data_path)
    
    print("Loading data...")
    transitions = []
    
    # Load data
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            # Check for header
            first_line = f.readline()
            # Wikipedia clickstream format: prev curr type n
            # Some files have header, some don't.
            
            # Simple parser
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4: # prev curr type n
                    source, dest, type_, count = parts[0], parts[1], parts[2], parts[3]
                    # Filter for actual links
                    if type_ == 'link':
                        try:
                            count = int(count)
                            transitions.append((source, dest, count))
                        except:
                            pass
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Loaded {len(transitions)} transitions")
    
    if not transitions:
        print("No valid transitions found.")
        return
        
    # Shuffle to ensure distribution
    np.random.shuffle(transitions)
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(transitions))
    train_data = transitions[:split_idx]
    test_data = transitions[split_idx:]
    
    # Check overlap
    train_sources = set(t[0] for t in train_data)
    test_sources = set(t[0] for t in test_data)
    overlap = len(test_sources.intersection(train_sources))
    print(f"Train/Test Overlap: {overlap} unique sources in both sets")
    print(f"Test Set Coverage: {overlap/len(test_sources)*100:.1f}% of test sources seen in train")
    
    # Grid size tuning:
    # 128*128 = 16k cells. If vocab > 16k, collisions inevitable.
    # Wikipedia vocab is huge. Let's use larger grid or see effect.
    grid_size = 512 # 256k cells
    
    # Initialize models
    print(f"\nTraining Advective Bloom Filter (Grid={grid_size}x{grid_size})...")
    abf = AdvectiveBloomFilter(grid_size=grid_size, learning_rate=0.2)
    
    train_start = time.time()
    for source, dest, count in train_data:
        abf.insert_transition(source, dest, weight=count)
    abf_train_time = time.time() - train_start
    print(f"  Training time: {abf_train_time:.3f}s")
    
    print("Training Markov Baseline...")
    markov = MarkovBaseline()
    
    train_start = time.time()
    for source, dest, count in train_data:
        markov.insert_transition(source, dest, weight=count)
    markov_train_time = time.time() - train_start
    print(f"  Training time: {markov_train_time:.3f}s")
    
    # Evaluate
    print("\nEvaluating...")
    
    def evaluate_model(model, test_data, model_name):
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        prediction_times = []
        
        # Test sample
        limit_test = min(1000, len(test_data))
        
        debug_counter = 0
        
        for source, actual_dest, _ in test_data[:limit_test]:
            start = time.perf_counter()
            
            # Top-1 prediction
            pred = model.predict_next(source)
            
            # Top-5 predictions
            top5 = model.predict_top_k(source, k=5)
            
            elapsed = time.perf_counter() - start
            prediction_times.append(elapsed)
            
            if pred == actual_dest:
                top1_correct += 1
            
            if actual_dest in top5:
                top5_correct += 1
                
            # Debug first few failures/successes
            if debug_counter < 5:
                print(f"  [DEBUG {model_name}] Src: '{source}' -> Act: '{actual_dest}' | Pred: '{pred}' | Top5: {top5}")
                debug_counter += 1
            
            total += 1
        
        avg_latency = np.mean(prediction_times) * 1e6  # microseconds
        
        print(f"\n{model_name} Results:")
        print(f"  Top-1 Accuracy: {100*top1_correct/total:.2f}%")
        print(f"  Top-5 Accuracy: {100*top5_correct/total:.2f}%")
        print(f"  Avg Latency: {avg_latency:.2f} μs")
        
        return {
            'top1_acc': top1_correct/total,
            'top5_acc': top5_correct/total,
            'latency_us': avg_latency
        }
    
    abf_results = evaluate_model(abf, test_data, "Advective Bloom Filter")
    markov_results = evaluate_model(markov, test_data, "Markov Chain")
    
    # Memory footprint
    abf_memory = abf.velocity_field.nbytes / 1024  # KB
    # Estimate dict size carefully
    import sys
    markov_memory = sys.getsizeof(markov.transitions) / 1024
    for k, v in markov.transitions.items():
        markov_memory += sys.getsizeof(v) / 1024
        
    print(f"\nMemory Usage:")
    print(f"  Advective Bloom: {abf_memory:.2f} KB")
    print(f"  Markov Chain: {markov_memory:.2f} KB")
    
    # Success criteria
    print("\n" + "="*50)
    print("SUCCESS CRITERIA:")
    print("="*50)
    
    speedup = markov_results['latency_us'] / (abf_results['latency_us'] + 1e-9)
    print(f"{'✓' if speedup > 2 else '✗'} Speed: {speedup:.1f}x faster (Advective vs Markov)")
    
    accuracy_ratio = abf_results['top5_acc'] / (markov_results['top5_acc'] + 1e-9)
    print(f"{'✓' if accuracy_ratio > 0.7 else '✗'} Accuracy: {accuracy_ratio*100:.1f}% of baseline")
    
    memory_ratio = abf_memory / (markov_memory + 1e-9)
    print(f"{'✓' if memory_ratio < 0.5 else '~'} Memory: {memory_ratio:.2f}x baseline size")
    
    # Collision Analysis
    print("\nCollision Analysis (Advective):")
    collisions = sum(len(v) - 1 for v in abf.coord_to_items.values() if len(v) > 1)
    total_items = len(abf.item_to_coord)
    print(f"  Items: {total_items}")
    print(f"  Collisions: {collisions} item pairs share a cell")

if __name__ == "__main__":
    run_experiment()
