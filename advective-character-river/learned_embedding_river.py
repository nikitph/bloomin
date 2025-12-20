
import numpy as np
import time
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

class LearnedCharacterEmbedding:
    """
    Learn a 2D embedding where characters that often appear in similar contexts
    are positioned near each other in space.
    """
    
    def __init__(self, embedding_dim=2, context_window=3):
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        
        # Character to vector mapping (will be learned)
        self.char_to_vec = {}
        
        # For building co-occurrence matrix
        self.vocab = set()
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self.char_to_idx = {}
        
    def build_cooccurrence_matrix(self, text):
        """
        Build character co-occurrence statistics.
        """
        print("Building co-occurrence matrix...")
        
        # Collect vocabulary
        for char in text:
            self.vocab.add(char)
        
        self.vocab = sorted(list(self.vocab))
        self.char_to_idx = {char: i for i, char in enumerate(self.vocab)}
        
        # Count co-occurrences within context window
        for i in range(len(text)):
            center_char = text[i]
            
            # Look at surrounding characters
            start = max(0, i - self.context_window)
            end = min(len(text), i + self.context_window + 1)
            
            for j in range(start, end):
                if i != j:
                    context_char = text[j]
                    distance = abs(i - j)
                    weight = 1.0 / distance  # Closer = stronger signal
                    
                    self.cooccurrence[center_char][context_char] += weight
        
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  Total co-occurrence pairs: {sum(len(v) for v in self.cooccurrence.values())}")
    
    def learn_embedding_pca(self):
        """
        Use PCA on co-occurrence matrix to get 2D embedding.
        """
        print("Learning embedding via PCA...")
        
        # Build co-occurrence matrix
        n = len(self.vocab)
        cooc_matrix = np.zeros((n, n))
        
        for i, char_i in enumerate(self.vocab):
            for char_j, count in self.cooccurrence[char_i].items():
                j = self.char_to_idx[char_j]
                cooc_matrix[i, j] = count
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=self.embedding_dim)
        embeddings = pca.fit_transform(cooc_matrix)
        
        # Normalize to [0, 1] range for grid mapping
        embeddings = embeddings - embeddings.min(axis=0)
        # Avoid div by zero if max is min (unlikely with PCA but safe)
        if embeddings.max() > 0:
            embeddings = embeddings / embeddings.max(axis=0)
        
        # Store mapping
        for i, char in enumerate(self.vocab):
            self.char_to_vec[char] = embeddings[i]
        
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        return embeddings
    
    def visualize_embedding(self, save_path='char_embedding.png'):
        """
        Plot the learned 2D character space.
        """
        print("Visualizing embedding...")
        
        plt.figure(figsize=(12, 10))
        
        # Plot each character
        for char, vec in self.char_to_vec.items():
            x, y = vec[0], vec[1]
            
            # Color by type
            if char.isalpha():
                if char.lower() in 'aeiou':
                    color = 'red'  # Vowels
                else:
                    color = 'blue'  # Consonants
            elif char.isspace():
                color = 'green'
            elif char in '.,;:!?':
                color = 'orange'
            else:
                color = 'gray'
            
            plt.scatter(x, y, c=color, s=100, alpha=0.6)
            plt.text(x, y, char, fontsize=8, ha='center', va='center')
        
        plt.title("Learned Character Embedding Space")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha=0.3)
        
        # Legend (simplified to avoid messy handles logic without artist objects)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Vowels', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Consonants', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Whitespace', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Punctuation', markersize=10),
        ]
        plt.legend(handles=legend_elements)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()


class StochasticRiverWithLearnedEmbedding:
    """
    Advective flow field on LEARNED character space (not ASCII).
    """
    
    def __init__(self, embedding, grid_size=64):
        self.embedding = embedding
        self.grid_size = grid_size
        
        # Mean velocity field
        self.mean_velocity = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
        
        # Covariance (simplified: just diagonal variances)
        self.var_x = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.var_y = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Welford accumulators
        self.count = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.M2_x = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.M2_y = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Reverse mapping: grid cell -> character
        self.grid_to_chars = defaultdict(list)
    
    def _char_to_grid(self, char):
        """Map character to grid coordinates using learned embedding."""
        if char not in self.embedding.char_to_vec:
            # Unknown character -> random position
            return (self.grid_size // 2, self.grid_size // 2)
        
        vec = self.embedding.char_to_vec[char]
        x = int(vec[0] * (self.grid_size - 1))
        y = int(vec[1] * (self.grid_size - 1))
        
        self.grid_to_chars[(x, y)].append(char)
        
        return (x, y)
    
    def train(self, text):
        """
        Learn flow field on the embedded space.
        """
        print("Training flow field on learned embedding...")
        
        # Batch cache positions for speed
        # But we need Welford loop, so standard iteration is fine
        
        for i in range(len(text) - 1):
            current_char = text[i]
            next_char = text[i + 1]
            
            x1, y1 = self._char_to_grid(current_char)
            x2, y2 = self._char_to_grid(next_char)
            
            # Velocity vector in embedding space
            dx = x2 - x1
            dy = y2 - y1
            
            # Welford update
            n = self.count[x1, y1]
            
            delta_x = dx - self.mean_velocity[x1, y1, 0]
            delta_y = dy - self.mean_velocity[x1, y1, 1]
            
            self.mean_velocity[x1, y1, 0] += delta_x / (n + 1)
            self.mean_velocity[x1, y1, 1] += delta_y / (n + 1)
            
            delta2_x = dx - self.mean_velocity[x1, y1, 0]
            delta2_y = dy - self.mean_velocity[x1, y1, 1]
            
            self.M2_x[x1, y1] += delta_x * delta2_x
            self.M2_y[x1, y1] += delta_y * delta2_y
            
            self.count[x1, y1] += 1
        
        # Compute final variances
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.count[x, y] > 1:
                    self.var_x[x, y] = self.M2_x[x, y] / self.count[x, y]
                    self.var_y[x, y] = self.M2_y[x, y] / self.count[x, y]
    
    def predict_next_stochastic(self, char, num_samples=50):
        """
        Sample from flow distribution to predict next character.
        """
        x, y = self._char_to_grid(char)
        
        mean_dx = self.mean_velocity[x, y, 0]
        mean_dy = self.mean_velocity[x, y, 1]
        
        std_dx = np.sqrt(max(self.var_x[x, y], 1e-6))
        std_dy = np.sqrt(max(self.var_y[x, y], 1e-6))
        
        # Sample velocities
        sampled_dx = np.random.normal(mean_dx, std_dx, num_samples)
        sampled_dy = np.random.normal(mean_dy, std_dy, num_samples)
        
        # Find characters at predicted locations
        predictions = []
        for dx, dy in zip(sampled_dx, sampled_dy):
            pred_x = int(np.clip(x + dx, 0, self.grid_size - 1))
            pred_y = int(np.clip(y + dy, 0, self.grid_size - 1))
            
            # Find nearest character(s) at this grid cell
            candidates = self.grid_to_chars.get((pred_x, pred_y), [])
            
            if candidates:
                predictions.append(np.random.choice(candidates))
            else:
                # Search nearby cells
                for radius in range(1, 5):
                    found = False
                    for dx_search in range(-radius, radius+1):
                        for dy_search in range(-radius, radius+1):
                            check_x = np.clip(pred_x + dx_search, 0, self.grid_size - 1)
                            check_y = np.clip(pred_y + dy_search, 0, self.grid_size - 1)
                            
                            candidates = self.grid_to_chars.get((check_x, check_y), [])
                            if candidates:
                                predictions.append(np.random.choice(candidates))
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
        
        # Count predictions
        from collections import Counter
        counts = Counter(predictions)
        total = len(predictions) if predictions else 1
        
        return [(char, count/total) for char, count in counts.most_common()]
    
    def predict_next_best(self, char):
        """Return most likely next character."""
        predictions = self.predict_next_stochastic(char, num_samples=100)
        if predictions:
            return predictions[0][0]
        return ' '


def run_learned_embedding_experiment():
    """
    Full pipeline: Learn embeddings, train flow field, evaluate.
    """
    
    print("="*60)
    print("LEARNED EMBEDDING + STOCHASTIC FLOW EXPERIMENT")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading data...")
    try:
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()[:100000]
    except FileNotFoundError:
        text = "To be or not to be " * 5000
    
    split = int(0.8 * len(text))
    train_text = text[:split]
    test_text = text[split:]
    
    # Learn embedding
    print("\n[2/4] Learning character embedding...")
    embedding = LearnedCharacterEmbedding(embedding_dim=2, context_window=5)
    embedding.build_cooccurrence_matrix(train_text)
    
    # Choose method: PCA (fast)
    embedding.learn_embedding_pca()
    
    try:
        embedding.visualize_embedding('char_embedding.png')
    except Exception as e:
        print(f"Skipping visualization: {e}")
    
    # Train flow field
    print("\n[3/4] Training stochastic flow field...")
    # Use grid_size 64 for decent resolution
    river = StochasticRiverWithLearnedEmbedding(embedding, grid_size=64)
    
    start = time.time()
    river.train(train_text)
    train_time = time.time() - start
    print(f"  Trained in {train_time:.3f}s")
    
    # Evaluate
    print("\n[4/4] Evaluating...")
    
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    
    prediction_times = []
    
    # Test on random samples
    # Ensure chars are in embedding
    valid_test_indices = [i for i in range(len(test_text)-1) if test_text[i] in embedding.vocab]
    
    if len(valid_test_indices) > 1000:
        samples = np.random.choice(valid_test_indices, 1000, replace=False)
    else:
        samples = valid_test_indices
        
    for pos in samples:
        current_char = test_text[pos]
        actual_next = test_text[pos + 1]
        
        start = time.perf_counter()
        predictions = river.predict_next_stochastic(current_char, num_samples=100)
        elapsed = time.perf_counter() - start
        prediction_times.append(elapsed)
        
        if predictions:
            # Top-1
            if predictions[0][0] == actual_next:
                correct_top1 += 1
            
            # Top-3
            top3 = [p[0] for p in predictions[:3]]
            if actual_next in top3:
                correct_top3 += 1
            
            # Top-5
            top5 = [p[0] for p in predictions[:5]]
            if actual_next in top5:
                correct_top5 += 1
        
        total += 1
        
    if total == 0:
        print("No valid test samples.")
        return
        
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nAccuracy:")
    print(f"  Top-1: {100*correct_top1/total:.2f}%")
    print(f"  Top-3: {100*correct_top3/total:.2f}%")
    print(f"  Top-5: {100*correct_top5/total:.2f}%")
    
    print(f"\nPerformance:")
    print(f"  Avg Latency: {np.mean(prediction_times)*1e6:.2f} μs")
    print(f"  Throughput: {1/np.mean(prediction_times):.0f} pred/sec")
    
    memory_kb = (river.mean_velocity.nbytes + 
                 river.var_x.nbytes + 
                 river.var_y.nbytes) / 1024
    print(f"  Memory: {memory_kb:.2f} KB")
    
    # Qualitative demo
    print("\n" + "="*60)
    print("TEXT GENERATION DEMO")
    print("="*60)
    
    def generate_text(river, seed, length=100):
        result = seed
        current = seed[-1]
        
        for _ in range(length):
            next_char = river.predict_next_best(current)
            result += next_char
            current = next_char
        
        return result
    
    seed = "To be or not to be"
    print(f"\nSeed: '{seed}'")
    try:
        print(f"Generated: {generate_text(river, seed, 100)}")
    except Exception as e:
        print(f"Generation failed: {e}")
    
    return {
        'top1': correct_top1/total,
        'top3': correct_top3/total,
        'top5': correct_top5/total,
        'latency_us': np.mean(prediction_times)*1e6,
        'memory_kb': memory_kb
    }


if __name__ == "__main__":
    results = run_learned_embedding_experiment()
