"""
Experiment D: Free-Will / Novel Action Test

Tests whether agent generates valid novel actions during dreaming
that were not in the training set.

Hypothesis: Fraction of novel actions that are valid/useful ≥10%
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import REWAMemory, ToposLayer, SemanticRG, AGIController
from datasets import create_hidden_rule_dataset


class FreeWillExperiment:
    """
    Free-Will / Novel Action Test
    
    Agent dreams and generates novel action combinations.
    Test: Are the novel actions valid and useful?
    """
    
    def __init__(
        self,
        n_training_actions: int = 500,
        n_dream_cycles: int = 100,
        dreams_per_cycle: int = 20,
        seed: int = 42
    ):
        self.n_training_actions = n_training_actions
        self.n_dream_cycles = n_dream_cycles
        self.dreams_per_cycle = dreams_per_cycle
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize modules
        self.rewa_memory = REWAMemory(K=32, L=64, seed=seed)
        self.topos_layer = ToposLayer(kl_threshold=0.1)
        self.semantic_rg = SemanticRG()
        self.controller = AGIController(
            temperature=0.2,  # Higher temperature for exploration
            query_budget=dreams_per_cycle,
            seed=seed
        )
        
        # Action space
        self.action_primitives = {
            'move': ['forward', 'backward', 'left', 'right'],
            'interact': ['pick', 'place', 'push', 'pull'],
            'observe': ['look', 'listen', 'touch']
        }
        
        # Valid action combinations (ground truth)
        self.valid_combinations = self._define_valid_combinations()
        
        # Training set
        self.training_actions: Set[str] = set()
        
        # Results
        self.results = {
            'novel_actions': [],
            'valid_novel_actions': [],
            'novelty_rate': [],
            'validity_rate': []
        }
    
    def _define_valid_combinations(self) -> Set[str]:
        """
        Define valid action combinations (physics/logic constraints).
        
        Example rules:
        - Can't pick and place simultaneously
        - Must move before interacting
        - Can observe while moving
        """
        valid = set()
        
        # Single actions (always valid)
        for category, actions in self.action_primitives.items():
            for action in actions:
                valid.add(action)
        
        # Valid combinations
        # Move + Observe
        for move in self.action_primitives['move']:
            for observe in self.action_primitives['observe']:
                valid.add(f"{move}+{observe}")
        
        # Move + Interact (sequential)
        for move in self.action_primitives['move']:
            for interact in self.action_primitives['interact']:
                valid.add(f"{move}>{interact}")
        
        # Observe + Interact
        for observe in self.action_primitives['observe']:
            for interact in self.action_primitives['interact']:
                valid.add(f"{observe}+{interact}")
        
        return valid
    
    def action_to_features(self, action_str: str) -> np.ndarray:
        """Convert action string to feature vector"""
        # Simple encoding: hash action string to feature space
        hash_val = hash(action_str) % (2**32)
        self.rng.seed(hash_val)
        features = self.rng.randn(64)
        features = features / (np.linalg.norm(features) + 1e-10)
        
        # Reset RNG
        self.rng.seed(self.seed)
        
        return features
    
    def generate_training_actions(self) -> List[str]:
        """Generate training set of actions (subset of valid actions)"""
        all_valid = list(self.valid_combinations)
        
        # Sample subset for training
        n_train = min(self.n_training_actions, len(all_valid))
        training_actions = self.rng.choice(all_valid, size=n_train, replace=False)
        
        return list(training_actions)
    
    def load_training_actions(self, training_actions: List[str]):
        """Load training actions into memory"""
        print(f"Loading {len(training_actions)} training actions...")
        
        for action_str in tqdm(training_actions):
            features = self.action_to_features(action_str)
            ws = self.rewa_memory.extract_witnesses(features, item_id=action_str)
            ws.metadata = {'action': action_str, 'type': 'training'}
            self.rewa_memory.store(ws)
            self.training_actions.add(action_str)
        
        print(f"Loaded {self.rewa_memory.get_statistics()['num_items']} actions")
    
    def dream_novel_actions(self) -> List[str]:
        """Generate novel actions through dreaming"""
        novel_actions = []
        
        print(f"\nDreaming for {self.n_dream_cycles} cycles...")
        for cycle in tqdm(range(self.n_dream_cycles)):
            # Sample from manifold
            sampled_ws = self.rewa_memory.sample_from_manifold(
                n_samples=self.dreams_per_cycle
            )
            
            # For each sample, try to decode to action
            for ws in sampled_ws:
                # Find nearest neighbor
                neighbors = self.rewa_memory.retrieve(ws, k=3)
                
                if len(neighbors) >= 2:
                    # Combine/interpolate actions
                    action1 = neighbors[0][0]
                    action2 = neighbors[1][0]
                    
                    # Simple combination heuristic
                    if '+' not in action1 and '+' not in action2:
                        novel_action = f"{action1}+{action2}"
                    elif '>' not in action1 and '>' not in action2:
                        novel_action = f"{action1}>{action2}"
                    else:
                        continue
                    
                    # Check if novel
                    if novel_action not in self.training_actions:
                        novel_actions.append(novel_action)
        
        return novel_actions
    
    def evaluate_novel_actions(self, novel_actions: List[str]) -> Dict:
        """Evaluate validity of novel actions"""
        valid_novel = []
        
        for action in novel_actions:
            if action in self.valid_combinations:
                valid_novel.append(action)
        
        total_novel = len(novel_actions)
        total_valid_novel = len(valid_novel)
        
        novelty_rate = total_novel / max(len(self.training_actions), 1)
        validity_rate = total_valid_novel / max(total_novel, 1)
        
        return {
            'total_novel': total_novel,
            'total_valid_novel': total_valid_novel,
            'novelty_rate': novelty_rate,
            'validity_rate': validity_rate,
            'novel_actions': novel_actions,
            'valid_novel_actions': valid_novel
        }
    
    def run(self) -> Dict:
        """Run full free-will experiment"""
        print(f"\n{'='*60}")
        print(f"Experiment D: Free-Will / Novel Action Test")
        print(f"Seed: {self.seed}")
        print(f"{'='*60}\n")
        
        # Generate and load training actions
        print("Generating training actions...")
        training_actions = self.generate_training_actions()
        self.load_training_actions(training_actions)
        
        print(f"\nTotal valid actions: {len(self.valid_combinations)}")
        print(f"Training actions: {len(self.training_actions)}")
        print(f"Held-out valid actions: {len(self.valid_combinations) - len(self.training_actions)}")
        
        # Dream novel actions
        novel_actions = self.dream_novel_actions()
        
        # Evaluate
        print(f"\nEvaluating {len(novel_actions)} novel actions...")
        eval_results = self.evaluate_novel_actions(novel_actions)
        
        # Store results
        self.results['novel_actions'] = novel_actions[:100]  # Store subset
        self.results['valid_novel_actions'] = eval_results['valid_novel_actions'][:100]
        self.results['novelty_rate'] = [eval_results['novelty_rate']]
        self.results['validity_rate'] = [eval_results['validity_rate']]
        
        # Print summary
        print(f"\n{'='*60}")
        print("Free-Will Test Complete!")
        print(f"{'='*60}")
        print(f"Novel Actions Generated: {eval_results['total_novel']}")
        print(f"Valid Novel Actions: {eval_results['total_valid_novel']}")
        print(f"Novelty Rate: {eval_results['novelty_rate']:.2%}")
        print(f"Validity Rate: {eval_results['validity_rate']:.2%}")
        
        if eval_results['validity_rate'] >= 0.10:
            print("✓ SUCCESS: Achieved ≥10% validity threshold")
        else:
            print("✗ BELOW THRESHOLD: Did not achieve 10% validity")
        
        # Show examples
        if len(eval_results['valid_novel_actions']) > 0:
            print(f"\nExample valid novel actions:")
            for i, action in enumerate(eval_results['valid_novel_actions'][:5]):
                print(f"  {i+1}. {action}")
        
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
        """Plot results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rates
        rates = [self.results['novelty_rate'][0], self.results['validity_rate'][0]]
        labels = ['Novelty Rate', 'Validity Rate']
        colors = ['steelblue', 'forestgreen']
        
        axes[0].bar(labels, rates, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Rate')
        axes[0].set_title('Novel Action Generation')
        axes[0].set_ylim([0, max(rates) * 1.2])
        axes[0].axhline(0.10, color='red', linestyle='--', 
                       label='Target: 10% validity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Action counts
        total_novel = len(self.results['novel_actions'])
        valid_novel = len(self.results['valid_novel_actions'])
        invalid_novel = total_novel - valid_novel
        
        axes[1].bar(['Valid Novel', 'Invalid Novel'], 
                   [valid_novel, invalid_novel],
                   color=['forestgreen', 'coral'],
                   alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Novel Action Breakdown')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")


def run_experiment(seed: int = 42, output_dir: str = '../logs') -> Dict:
    """Run free-will experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    experiment = FreeWillExperiment(
        n_training_actions=500,
        n_dream_cycles=100,
        dreams_per_cycle=20,
        seed=seed
    )
    
    results = experiment.run()
    
    experiment.save_results(f"{output_dir}/results_freewill_test_seed{seed}.json")
    experiment.plot_results(f"{output_dir}/plot_freewill_test_seed{seed}.png")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Free-Will Test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='../logs', help='Output directory')
    
    args = parser.parse_args()
    
    results = run_experiment(seed=args.seed, output_dir=args.output_dir)
    
    print("\n✓ Experiment complete!")
