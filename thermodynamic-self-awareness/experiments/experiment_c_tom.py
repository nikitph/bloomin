"""
Experiment C: Theory-of-Mind Test

Tests whether agent A can model agent B's knowledge state and predict
what agent B knows or doesn't know.

Hypothesis: Agent A predicts agent B's ignorance with accuracy ≥85%
and well-calibrated confidence.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import REWAMemory, ToposLayer, WitnessSet


class TheoryOfMindExperiment:
    """
    Theory-of-Mind Test
    
    Agent A observes events X and Y.
    Agent B observes only event Y.
    Test: Can A predict that B doesn't know X?
    """
    
    def __init__(
        self,
        n_shared_events: int = 500,
        n_a_only_events: int = 100,
        n_b_only_events: int = 100,
        n_test_queries: int = 50,
        seed: int = 42
    ):
        self.n_shared_events = n_shared_events
        self.n_a_only_events = n_a_only_events
        self.n_b_only_events = n_b_only_events
        self.n_test_queries = n_test_queries
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Two agents
        self.agent_a = REWAMemory(K=32, L=64, seed=seed)
        self.agent_b = REWAMemory(K=32, L=64, seed=seed + 1)
        
        # Results
        self.results = {
            'accuracies': [],
            'confidences': [],
            'predictions': [],
            'true_labels': []
        }
    
    def generate_events(self) -> Tuple[List, List, List]:
        """
        Generate events observed by agents.
        
        Returns:
            (shared_events, a_only_events, b_only_events)
        """
        # Shared events (both agents observe)
        shared_events = []
        for i in range(self.n_shared_events):
            features = self.rng.randn(64)
            features = features / (np.linalg.norm(features) + 1e-10)
            shared_events.append(features)
        
        # Events only A observes
        a_only_events = []
        for i in range(self.n_a_only_events):
            features = self.rng.randn(64)
            features = features / (np.linalg.norm(features) + 1e-10)
            a_only_events.append(features)
        
        # Events only B observes
        b_only_events = []
        for i in range(self.n_b_only_events):
            features = self.rng.randn(64)
            features = features / (np.linalg.norm(features) + 1e-10)
            b_only_events.append(features)
        
        return shared_events, a_only_events, b_only_events
    
    def load_observations(
        self,
        shared_events: List,
        a_only_events: List,
        b_only_events: List
    ):
        """Load observations into agent memories"""
        print("Loading observations into Agent A...")
        # Agent A sees shared + A-only events
        for i, features in enumerate(shared_events):
            ws = self.agent_a.extract_witnesses(features, item_id=f"shared_{i}")
            ws.metadata = {'type': 'shared'}
            self.agent_a.store(ws)
        
        for i, features in enumerate(a_only_events):
            ws = self.agent_a.extract_witnesses(features, item_id=f"a_only_{i}")
            ws.metadata = {'type': 'a_only'}
            self.agent_a.store(ws)
        
        print("Loading observations into Agent B...")
        # Agent B sees shared + B-only events
        for i, features in enumerate(shared_events):
            ws = self.agent_b.extract_witnesses(features, item_id=f"shared_{i}")
            ws.metadata = {'type': 'shared'}
            self.agent_b.store(ws)
        
        for i, features in enumerate(b_only_events):
            ws = self.agent_b.extract_witnesses(features, item_id=f"b_only_{i}")
            ws.metadata = {'type': 'b_only'}
            self.agent_b.store(ws)
        
        print(f"Agent A memory: {self.agent_a.get_statistics()['num_items']} items")
        print(f"Agent B memory: {self.agent_b.get_statistics()['num_items']} items")
    
    def agent_a_predicts_b_knowledge(
        self,
        query_event: np.ndarray,
        event_type: str
    ) -> Tuple[bool, float]:
        """
        Agent A predicts whether Agent B knows about the query event.
        
        Args:
            query_event: Event features
            event_type: 'shared', 'a_only', or 'b_only'
            
        Returns:
            (predicted_knows, confidence)
        """
        # Create witness for query
        query_ws = self.agent_a.extract_witnesses(query_event, item_id="query")
        
        # Agent A checks its own memory
        a_neighbors = self.agent_a.retrieve(query_ws, k=5)
        
        # Simple heuristic: if A has seen similar events, check if they're shared
        # In a more sophisticated implementation, A would maintain explicit model of B
        shared_count = 0
        total_count = 0
        
        for item_id, stats in a_neighbors:
            if item_id in self.agent_a.memory:
                item_type = self.agent_a.memory[item_id].metadata.get('type', 'unknown')
                total_count += 1
                if item_type == 'shared':
                    shared_count += 1
        
        if total_count > 0:
            # Probability B knows = fraction of shared neighbors
            prob_b_knows = shared_count / total_count
        else:
            prob_b_knows = 0.5  # Uncertain
        
        predicted_knows = prob_b_knows > 0.5
        confidence = max(prob_b_knows, 1 - prob_b_knows)
        
        return predicted_knows, confidence
    
    def test_theory_of_mind(
        self,
        test_events: List[Tuple[np.ndarray, str]]
    ) -> Dict:
        """
        Test Agent A's ability to predict Agent B's knowledge.
        
        Args:
            test_events: List of (event_features, event_type) tuples
            
        Returns:
            Dictionary with accuracy and confidence metrics
        """
        correct = 0
        predictions = []
        confidences = []
        true_labels = []
        
        for event_features, event_type in tqdm(test_events, desc="Testing ToM"):
            # Ground truth: does B actually know this event?
            query_ws = self.agent_b.extract_witnesses(event_features, item_id="query_b")
            b_neighbors = self.agent_b.retrieve(query_ws, k=1)
            
            # B knows if it has high overlap with something in memory
            if len(b_neighbors) > 0:
                overlap = b_neighbors[0][1]['overlap']
                b_actually_knows = overlap > 0.7  # Threshold for "knowing"
            else:
                b_actually_knows = False
            
            # Agent A's prediction
            predicted_knows, confidence = self.agent_a_predicts_b_knowledge(
                event_features, event_type
            )
            
            # Check correctness
            if predicted_knows == b_actually_knows:
                correct += 1
            
            predictions.append(predicted_knows)
            confidences.append(confidence)
            true_labels.append(b_actually_knows)
        
        accuracy = correct / len(test_events)
        
        return {
            'accuracy': accuracy,
            'mean_confidence': np.mean(confidences),
            'predictions': predictions,
            'confidences': confidences,
            'true_labels': true_labels
        }
    
    def run(self) -> Dict:
        """Run full theory-of-mind experiment"""
        print(f"\n{'='*60}")
        print(f"Experiment C: Theory-of-Mind Test")
        print(f"Seed: {self.seed}")
        print(f"{'='*60}\n")
        
        # Generate events
        print("Generating events...")
        shared_events, a_only_events, b_only_events = self.generate_events()
        
        # Load observations
        self.load_observations(shared_events, a_only_events, b_only_events)
        
        # Create test queries
        print(f"\nCreating {self.n_test_queries} test queries...")
        test_events = []
        
        # Test on A-only events (B should NOT know)
        n_a_test = min(self.n_test_queries // 2, len(a_only_events))
        for i in range(n_a_test):
            test_events.append((a_only_events[i], 'a_only'))
        
        # Test on shared events (B SHOULD know)
        n_shared_test = self.n_test_queries - n_a_test
        for i in range(n_shared_test):
            idx = self.rng.randint(len(shared_events))
            test_events.append((shared_events[idx], 'shared'))
        
        # Run test
        results = self.test_theory_of_mind(test_events)
        
        # Store results
        self.results['accuracies'] = [results['accuracy']]
        self.results['confidences'] = results['confidences']
        self.results['predictions'] = results['predictions']
        self.results['true_labels'] = results['true_labels']
        
        # Print summary
        print(f"\n{'='*60}")
        print("Theory-of-Mind Test Complete!")
        print(f"{'='*60}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Mean Confidence: {results['mean_confidence']:.2%}")
        
        if results['accuracy'] >= 0.85:
            print("✓ SUCCESS: Achieved ≥85% accuracy threshold")
        else:
            print("✗ BELOW THRESHOLD: Did not achieve 85% accuracy")
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save results to JSON"""
        results_serializable = {
            k: [bool(v) if isinstance(v, (bool, np.bool_)) else
                float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for v in vals] if isinstance(vals, list) else vals
            for k, vals in self.results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def plot_results(self, output_path: str):
        """Plot results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence distribution
        axes[0].hist(self.results['confidences'], bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Confidence Distribution')
        axes[0].axvline(np.mean(self.results['confidences']), 
                       color='red', linestyle='--', 
                       label=f'Mean: {np.mean(self.results["confidences"]):.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy by prediction type
        true_labels = self.results['true_labels']
        predictions = self.results['predictions']
        
        correct_knows = sum(1 for t, p in zip(true_labels, predictions) 
                          if t == True and p == True)
        total_knows = sum(1 for t in true_labels if t == True)
        
        correct_not_knows = sum(1 for t, p in zip(true_labels, predictions) 
                               if t == False and p == False)
        total_not_knows = sum(1 for t in true_labels if t == False)
        
        acc_knows = correct_knows / total_knows if total_knows > 0 else 0
        acc_not_knows = correct_not_knows / total_not_knows if total_not_knows > 0 else 0
        
        axes[1].bar(['B Knows', 'B Doesn\'t Know'], 
                   [acc_knows, acc_not_knows],
                   alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy by Knowledge State')
        axes[1].set_ylim([0, 1])
        axes[1].axhline(0.85, color='red', linestyle='--', label='Target: 85%')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")


def run_experiment(seed: int = 42, output_dir: str = '../logs') -> Dict:
    """Run theory-of-mind experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    experiment = TheoryOfMindExperiment(
        n_shared_events=500,
        n_a_only_events=100,
        n_b_only_events=100,
        n_test_queries=50,
        seed=seed
    )
    
    results = experiment.run()
    
    experiment.save_results(f"{output_dir}/results_tom_test_seed{seed}.json")
    experiment.plot_results(f"{output_dir}/plot_tom_test_seed{seed}.png")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Theory-of-Mind Test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='../logs', help='Output directory')
    
    args = parser.parse_args()
    
    results = run_experiment(seed=args.seed, output_dir=args.output_dir)
    
    print("\n✓ Experiment complete!")
