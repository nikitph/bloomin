"""
Experiment B: Mirror Test (Self-Recognition)

Tests whether the agent can identify its own witness distribution
and discriminate self from other agents.

Hypothesis: Agent achieves ≥90% accuracy in self-recognition with
well-calibrated confidence scores.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import REWAMemory, ToposLayer, WitnessSet
from datasets import create_hidden_rule_dataset


class MirrorTestExperiment:
    """
    Mirror Test: Self-Recognition Experiment
    
    Tests whether agent can identify its own witness patterns.
    """
    
    def __init__(
        self,
        n_self_examples: int = 100,
        n_other_examples: int = 900,
        n_test_queries: int = 100,
        seed: int = 42
    ):
        self.n_self_examples = n_self_examples
        self.n_other_examples = n_other_examples
        self.n_test_queries = n_test_queries
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize modules
        self.rewa_memory = REWAMemory(K=32, L=64, seed=seed)
        self.topos_layer = ToposLayer(kl_threshold=0.1)
        
        # Results
        self.results = {
            'accuracies': [],
            'confidences': [],
            'true_labels': [],
            'predicted_labels': [],
            'predicted_probs': []
        }
    
    def create_self_signature(self) -> np.ndarray:
        """
        Create a unique signature for 'self' agent.
        This represents the agent's characteristic witness pattern.
        """
        # Create a distinctive pattern in a specific subspace
        signature = self.rng.randn(64)
        
        # Add strong signal in specific dimensions (agent's "identity")
        signature[0:10] = 2.0  # Strong positive signal
        signature[10:20] = -2.0  # Strong negative signal
        
        return signature / (np.linalg.norm(signature) + 1e-10)
    
    def generate_dataset(self) -> Tuple[List, List]:
        """
        Generate dataset with self and other examples.
        
        Returns:
            (self_examples, other_examples)
        """
        self_signature = self.create_self_signature()
        
        # Generate self examples (with signature + noise)
        self_examples = []
        for i in range(self.n_self_examples):
            # Add noise to signature
            features = self_signature + self.rng.randn(64) * 0.3
            features = features / (np.linalg.norm(features) + 1e-10)
            
            ws = self.rewa_memory.extract_witnesses(features, item_id=f"self_{i}")
            ws.metadata = {'label': 'self', 'type': 'self'}
            self_examples.append(ws)
        
        # Generate other examples (random, no signature)
        other_examples = []
        for i in range(self.n_other_examples):
            features = self.rng.randn(64)
            features = features / (np.linalg.norm(features) + 1e-10)
            
            ws = self.rewa_memory.extract_witnesses(features, item_id=f"other_{i}")
            ws.metadata = {'label': 'other', 'type': 'other'}
            other_examples.append(ws)
        
        return self_examples, other_examples
    
    def load_training_data(self, self_examples: List, other_examples: List):
        """Load examples into memory"""
        print(f"Loading {len(self_examples)} self examples...")
        for ws in self_examples:
            self.rewa_memory.store(ws)
        
        print(f"Loading {len(other_examples)} other examples...")
        for ws in other_examples:
            self.rewa_memory.store(ws)
        
        print(f"Total items in memory: {self.rewa_memory.get_statistics()['num_items']}")
    
    def test_self_recognition(self, test_examples: List[WitnessSet]) -> Dict:
        """
        Test self-recognition on held-out examples.
        
        Args:
            test_examples: List of test witness sets with ground truth labels
            
        Returns:
            Dictionary with accuracy and confidence metrics
        """
        correct = 0
        predictions = []
        confidences = []
        true_labels = []
        
        for test_ws in tqdm(test_examples, desc="Testing"):
            # Retrieve nearest neighbors
            neighbors = self.rewa_memory.retrieve(test_ws, k=10)
            
            # Count self vs other in neighbors
            self_count = 0
            other_count = 0
            
            for item_id, stats in neighbors:
                if item_id in self.rewa_memory.memory:
                    label = self.rewa_memory.memory[item_id].metadata.get('label', 'other')
                    if label == 'self':
                        self_count += 1
                    else:
                        other_count += 1
            
            # Predict based on majority vote
            total = self_count + other_count
            if total > 0:
                self_prob = self_count / total
                predicted = 'self' if self_prob > 0.5 else 'other'
                confidence = max(self_prob, 1 - self_prob)
            else:
                predicted = 'other'
                confidence = 0.5
                self_prob = 0.5
            
            # Check correctness
            true_label = test_ws.metadata.get('label', 'other')
            if predicted == true_label:
                correct += 1
            
            predictions.append(predicted)
            confidences.append(confidence)
            true_labels.append(true_label)
            self.results['predicted_probs'].append(self_prob)
        
        accuracy = correct / len(test_examples)
        
        return {
            'accuracy': accuracy,
            'mean_confidence': np.mean(confidences),
            'predictions': predictions,
            'confidences': confidences,
            'true_labels': true_labels
        }
    
    def run(self) -> Dict:
        """Run full mirror test experiment"""
        print(f"\n{'='*60}")
        print(f"Experiment B: Mirror Test (Self-Recognition)")
        print(f"Seed: {self.seed}")
        print(f"{'='*60}\n")
        
        # Generate dataset
        print("Generating dataset...")
        self_examples, other_examples = self.generate_dataset()
        
        # Split self examples into train/test
        n_self_train = int(0.7 * len(self_examples))
        self_train = self_examples[:n_self_train]
        self_test = self_examples[n_self_train:]
        
        # Split other examples into train/test
        n_other_train = int(0.7 * len(other_examples))
        other_train = other_examples[:n_other_train]
        other_test = other_examples[n_other_train:]
        
        # Load training data
        self.load_training_data(self_train, other_train)
        
        # Test on held-out examples
        test_examples = self_test + other_test
        self.rng.shuffle(test_examples)
        test_examples = test_examples[:self.n_test_queries]
        
        print(f"\nTesting on {len(test_examples)} held-out examples...")
        results = self.test_self_recognition(test_examples)
        
        # Store results
        self.results['accuracies'] = [results['accuracy']]
        self.results['confidences'] = results['confidences']
        self.results['true_labels'] = results['true_labels']
        self.results['predicted_labels'] = results['predictions']
        
        # Print summary
        print(f"\n{'='*60}")
        print("Mirror Test Complete!")
        print(f"{'='*60}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Mean Confidence: {results['mean_confidence']:.2%}")
        
        if results['accuracy'] >= 0.9:
            print("✓ SUCCESS: Achieved ≥90% accuracy threshold")
        else:
            print("✗ BELOW THRESHOLD: Did not achieve 90% accuracy")
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save results to JSON"""
        results_serializable = {
            k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for v in vals] if isinstance(vals, list) else vals
            for k, vals in self.results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def plot_results(self, output_path: str):
        """Plot calibration and confusion matrix"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence histogram
        axes[0].hist(self.results['confidences'], bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Confidence Distribution')
        axes[0].axvline(np.mean(self.results['confidences']), 
                       color='red', linestyle='--', 
                       label=f'Mean: {np.mean(self.results["confidences"]):.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confusion matrix
        true_labels = self.results['true_labels']
        pred_labels = self.results['predicted_labels']
        
        # Compute confusion matrix
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 'self' and p == 'self')
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 'other' and p == 'self')
        tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 'other' and p == 'other')
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 'self' and p == 'other')
        
        conf_matrix = np.array([[tp, fn], [fp, tn]])
        
        im = axes[1].imshow(conf_matrix, cmap='Blues', aspect='auto')
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(['Self', 'Other'])
        axes[1].set_yticklabels(['Self', 'Other'])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = axes[1].text(j, i, conf_matrix[i, j],
                                   ha="center", va="center", color="black", fontsize=16)
        
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")


def run_experiment(seed: int = 42, output_dir: str = '../logs') -> Dict:
    """Run mirror test experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    experiment = MirrorTestExperiment(
        n_self_examples=100,
        n_other_examples=900,
        n_test_queries=100,
        seed=seed
    )
    
    results = experiment.run()
    
    experiment.save_results(f"{output_dir}/results_mirror_test_seed{seed}.json")
    experiment.plot_results(f"{output_dir}/plot_mirror_test_seed{seed}.png")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Mirror Test Experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='../logs', help='Output directory')
    
    args = parser.parse_args()
    
    results = run_experiment(seed=args.seed, output_dir=args.output_dir)
    
    print("\n✓ Experiment complete!")
