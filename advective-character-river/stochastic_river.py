
import numpy as np
import time
from collections import defaultdict
import os

class StochasticCharacterRiver:
    """Primitive 49: Flow field with uncertainty"""
    
    def __init__(self, grid_size=256):
        self.grid_size = grid_size
        
        # Mean velocity
        self.mean_velocity = np.zeros((grid_size, 2), dtype=np.float32)
        
        # Covariance matrix (store as 3 values: var_x, var_y, cov_xy)
        # Simplified: Just Variance X for 1D case, or Var X/Y for 2D?
        # User prompt uses 1D logic (velocity_x) but init defines 2D mean_velocity.
        # Let's align with the prompt's provided logic which is hybrid.
        # Prompt logic: 
        #   velocity_x = next_idx - current_idx (1D logic on indices)
        #   mean_velocity[idx, 0] updated with delta_x
        #   So strictly 1D logic mapped into 2D storage? 
        #   Let's stick to the prompt's 1D flow logic since grid_size=256 implies 1D array of chars.
        
        self.cov_xx = np.zeros(grid_size, dtype=np.float32)
        
        # For online updates (Welford's algorithm)
        self.count = np.zeros(grid_size, dtype=np.int32)
        self.M2_x = np.zeros(grid_size, dtype=np.float32)
        
    def _char_to_idx(self, char):
        return ord(char) % self.grid_size
    
    def train(self, text):
        """
        Learn flow field + uncertainty from text
        Uses Welford's online algorithm for numerical stability
        """
        for i in range(len(text) - 1):
            current_char = text[i]
            next_char = text[i + 1]
            
            current_idx = self._char_to_idx(current_char)
            next_idx = self._char_to_idx(next_char)
            
            # Velocity for this transition
            velocity_x = next_idx - current_idx
            
            # Handle wraparound
            if abs(velocity_x) > self.grid_size // 2:
                velocity_x = velocity_x - np.sign(velocity_x) * self.grid_size
            
            # Welford's online update for mean and variance
            n = self.count[current_idx]
            
            # Update mean
            delta_x = velocity_x - self.mean_velocity[current_idx, 0]
            self.mean_velocity[current_idx, 0] += delta_x / (n + 1)
            
            # Update M2 (for variance calculation)
            delta2_x = velocity_x - self.mean_velocity[current_idx, 0]
            self.M2_x[current_idx] += delta_x * delta2_x
            
            self.count[current_idx] += 1
        
        # Compute final covariance
        for idx in range(self.grid_size):
            if self.count[idx] > 1:
                self.cov_xx[idx] = self.M2_x[idx] / self.count[idx]
    
    def predict_next_stochastic(self, char, num_samples=10):
        """
        Sample from the distribution to get multiple predictions
        Returns list of (predicted_char, probability)
        """
        current_idx = self._char_to_idx(char)
        
        # Get distribution parameters
        mean_v = self.mean_velocity[current_idx, 0]
        var_v = max(self.cov_xx[current_idx], 1e-6)  # Avoid zero variance
        
        # Sample velocities
        sampled_velocities = np.random.normal(mean_v, np.sqrt(var_v), num_samples)
        
        # Apply each velocity
        predictions = []
        for v in sampled_velocities:
            # We must handle the float index and wrapping carefully.
            # Ideally round to nearest int? 
            # Prompt logic: predicted_idx = int(current_idx + v)
            
            predicted_idx = int(current_idx + v) % self.grid_size
            predicted_char = chr(predicted_idx % 256)
            predictions.append(predicted_char)
        
        # Count occurrences
        from collections import Counter
        counts = Counter(predictions)
        
        # Return sorted by frequency
        total = len(predictions)
        return [(char, count/total) for char, count in counts.most_common()]
    
    def predict_next_best(self, char):
        """Return single most likely next character"""
        predictions = self.predict_next_stochastic(char, num_samples=50)
        if predictions:
            return predictions[0][0]
        return ' '


def run_stochastic_experiment():
    """
    Test if uncertainty modeling fixes the accuracy problem
    """
    
    print("Loading training data...")
    try:
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()[:100000]
    except FileNotFoundError:
        # Just use dummy text if file not found (though previous step should have downloaded it)
        text = "To be or not to be " * 5000 
    
    split = int(0.8 * len(text))
    train_text = text[:split]
    test_text = text[split:]
    
    print(f"\nTraining Stochastic Character River...")
    river = StochasticCharacterRiver(grid_size=256)
    
    start = time.time()
    river.train(train_text)
    train_time = time.time() - start
    print(f"  Trained in {train_time:.3f}s")
    
    # ===== Evaluate =====
    print("\nEvaluating with stochastic sampling...")
    
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    prediction_times = []
    
    # Test on 1000 samples
    for _ in range(1000):
        pos = np.random.randint(0, len(test_text) - 1)
        current_char = test_text[pos]
        actual_next = test_text[pos + 1]
        
        # Time prediction
        start = time.perf_counter()
        predictions = river.predict_next_stochastic(current_char, num_samples=50)
        elapsed = time.perf_counter() - start
        prediction_times.append(elapsed)
        
        # Check top-1
        if predictions and predictions[0][0] == actual_next:
            correct_top1 += 1
        
        # Check top-3
        top3_chars = [p[0] for p in predictions[:3]]
        if actual_next in top3_chars:
            correct_top3 += 1
        
        total += 1
    
    accuracy_top1 = correct_top1 / total
    accuracy_top3 = correct_top3 / total
    avg_latency_us = np.mean(prediction_times) * 1e6
    
    print(f"\nStochastic Flow Results:")
    print(f"  Top-1 Accuracy: {accuracy_top1*100:.2f}%")
    print(f"  Top-3 Accuracy: {accuracy_top3*100:.2f}%")
    print(f"  Latency: {avg_latency_us:.2f} μs")
    
    # Memory
    memory_kb = (river.mean_velocity.nbytes + 
                 river.cov_xx.nbytes) / 1024
    
    print(f"  Memory: {memory_kb:.2f} KB")
    
    # ===== Qualitative Demo =====
    print("\n" + "="*60)
    print("QUALITATIVE DEMO")
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
    print(f"\nGenerated text:")
    try:
        print(generate_text(river, seed, 100))
    except Exception as e:
        print(f"Generation failed: {e}")
    
    return accuracy_top1, accuracy_top3, avg_latency_us


if __name__ == "__main__":
    run_stochastic_experiment()
