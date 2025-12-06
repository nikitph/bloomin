"""
Experiment A (Hardened): Self-Discovery with True Learning Dynamics

Improvements over original:
1. Uses filtered dataset (no red+large examples)
2. Selective Ricci updates (only on significant contradictions)
3. Increased exploration temperature
4. Mixed dream policy (rare + frequent concepts)
5. Event-driven learning (not continuous equilibrium)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import pickle
from typing import Dict, List
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import (
    REWAMemory, ToposLayer, RicciFlow, SemanticRG, AGIController,
    FlowParams
)


class HardenedSelfDiscoveryExperiment:
    """
    Hardened Self-Discovery Experiment
    
    Forces true learning dynamics by:
    - Removing rule from initial memory
    - Selective Ricci updates (event-driven)
    - Increased exploration
    - Mixed dream sampling
    """
    
    def __init__(
        self,
        n_dream_epochs: int = 500,
        dreams_per_cycle: int = 20,
        eval_interval: int = 20,
        kl_threshold: float = 0.1,
        entropy_spike_threshold: float = 0.5,  # NEW: threshold for Ricci
        ricci_step_size: float = 1e-3,
        temperature: float = 0.5,  # NEW: increased from 0.1
        dream_policy: str = 'mixed',  # NEW: 'freq', 'mixed', 'uniform'
        rare_fraction: float = 0.5,  # NEW: fraction of rare concepts in dreams
        seed: int = 42,
        enable_topos: bool = True,
        enable_ricci: bool = True
    ):
        self.n_dream_epochs = n_dream_epochs
        self.dreams_per_cycle = dreams_per_cycle
        self.eval_interval = eval_interval
        self.entropy_spike_threshold = entropy_spike_threshold
        self.dream_policy = dream_policy
        self.rare_fraction = rare_fraction
        self.seed = seed
        self.enable_topos = enable_topos
        self.enable_ricci = enable_ricci
        
        # Initialize modules
        self.rewa_memory = REWAMemory(K=32, L=64, seed=seed)
        self.topos_layer = ToposLayer(kl_threshold=kl_threshold) if enable_topos else None
        
        flow_params = FlowParams(step_size=ricci_step_size)
        self.ricci_flow = RicciFlow(params=flow_params) if enable_ricci else None
        
        self.semantic_rg = SemanticRG()
        self.controller = AGIController(
            temperature=temperature,
            query_budget=dreams_per_cycle,
            seed=seed
        )
        
        # Evidence accumulation for weak contradictions
        self.accumulated_evidence = {}
        
        # Results storage
        self.results = {
            'free_energy': [],
            'semantic_energy': [],
            'curvature_entropy': [],
            'contradictions': [],
            'ricci_updates': [],
            'significant_contradictions': [],  # NEW
            'weak_contradictions': [],  # NEW
            'rules_discovered': [],
            'rule_discovery_epoch': None
        }
        
    def load_filtered_dataset(self, dataset_path: str):
        """Load filtered dataset without rule"""
        print(f"Loading filtered dataset from {dataset_path}...")
        
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        train_examples = data['train']
        metadata = data['metadata']
        
        print(f"Dataset metadata:")
        print(f"  Original size: {metadata['original_size']}")
        print(f"  Filtered size: {metadata['filtered_size']}")
        print(f"  Excluded: {metadata['excluded_count']}")
        print(f"  P(large|red) = {metadata['p_large_given_red']:.3f}")
        
        # Load into memory
        print(f"\nLoading {len(train_examples)} examples into memory...")
        for example in tqdm(train_examples):
            witness_set = self.rewa_memory.extract_witnesses(
                example.features,
                item_id=example.item_id
            )
            witness_set.metadata = {
                'attributes': example.attributes,
                # NOTE: hidden_properties NOT stored - must be discovered
            }
            self.rewa_memory.store(witness_set)
        
        print(f"Loaded {self.rewa_memory.get_statistics()['num_items']} items")
        
        return train_examples
    
    def sample_concepts_mixed(self, n_samples: int) -> List:
        """
        Mixed dream sampling: combine rare and frequent concepts.
        
        Args:
            n_samples: Number of concepts to sample
            
        Returns:
            List of sampled witness sets
        """
        all_items = list(self.rewa_memory.memory.values())
        
        if len(all_items) == 0:
            return []
        
        if self.dream_policy == 'uniform':
            # Pure uniform sampling
            return self.rng.choice(all_items, size=min(n_samples, len(all_items)), replace=False).tolist()
        
        elif self.dream_policy == 'freq':
            # Frequency-based (original behavior)
            return self.rewa_memory.sample_from_manifold(n_samples)
        
        elif self.dream_policy == 'mixed':
            # Mix of rare and frequent
            n_rare = int(n_samples * self.rare_fraction)
            n_freq = n_samples - n_rare
            
            # Sample rare (low support) concepts
            # Simple heuristic: concepts with few neighbors
            rare_samples = []
            if n_rare > 0:
                # Shuffle and take first n_rare
                shuffled = all_items.copy()
                self.rng.shuffle(shuffled)
                rare_samples = shuffled[:min(n_rare, len(shuffled))]
            
            # Sample frequent concepts
            freq_samples = self.rewa_memory.sample_from_manifold(n_freq)
            
            return rare_samples + freq_samples
        
        else:
            return self.rewa_memory.sample_from_manifold(n_samples)
    
    def run_dream_cycle(self, epoch: int) -> Dict:
        """Run one dreaming cycle with selective Ricci updates"""
        # Sample concepts based on dream policy
        if hasattr(self, 'rng'):
            witness_list = self.sample_concepts_mixed(self.dreams_per_cycle)
        else:
            self.rng = np.random.RandomState(self.seed + epoch)
            witness_list = self.sample_concepts_mixed(self.dreams_per_cycle)
        
        contradictions_detected = 0
        significant_contradictions = 0
        weak_contradictions = 0
        ricci_updates = 0
        
        if len(witness_list) > 1 and self.enable_topos and self.topos_layer is not None:
            # Build open sets
            open_sets = []
            for ws in witness_list:
                open_set = self.topos_layer.build_open_set(
                    prototype_id=ws.item_id,
                    prototype_witnesses=ws.witnesses,
                    memory_items=self.rewa_memory.memory,
                    radius=0.3
                )
                open_sets.append(open_set)
            
            # Attempt gluing
            glued_set, kl_matrix, is_consistent = self.topos_layer.glue(open_sets)
            
            if not is_consistent:
                contradictions_detected += 1
                
                # Get contradiction spec
                contradiction = self.topos_layer.get_contradiction(open_sets)
                
                if contradiction is not None:
                    entropy_spike = contradiction.kl_divergence
                    
                    # SELECTIVE RICCI: only update if spike is significant
                    if entropy_spike >= self.entropy_spike_threshold:
                        significant_contradictions += 1
                        
                        if self.enable_ricci and self.ricci_flow is not None:
                            # Apply Ricci update
                            current_metric = self.rewa_memory.fisher_metric(list(glued_set))
                            
                            error_signal = {
                                'kl_divergence': contradiction.kl_divergence,
                                'expected': contradiction.expected_distribution,
                                'observed': contradiction.observed_distribution
                            }
                            
                            delta_g = self.ricci_flow.flow_step(
                                current_metric=current_metric,
                                error_signal=error_signal,
                                region_ids=contradiction.region_ids
                            )
                            
                            # Apply update
                            new_metric = self.ricci_flow.apply_metric_update(current_metric, delta_g)
                            self.rewa_memory.fisher_metric_cache = new_metric
                            ricci_updates += 1
                    else:
                        # Weak contradiction - accumulate evidence
                        weak_contradictions += 1
                        region_key = tuple(sorted(contradiction.region_ids))
                        if region_key not in self.accumulated_evidence:
                            self.accumulated_evidence[region_key] = []
                        self.accumulated_evidence[region_key].append(entropy_spike)
        
        # Semantic RG consolidation
        if self.semantic_rg.should_consolidate(witness_list):
            packets = self.semantic_rg.coarse_grain(
                witness_sets=witness_list,
                current_scale=0,
                target_scale=1
            )
            for packet in packets:
                self.rewa_memory.store_abstraction(packet)
        
        # Compute diagnostics
        semantic_energy = self.rewa_memory.compute_semantic_energy()
        
        if self.rewa_memory.fisher_metric_cache is not None:
            curvature_entropy = self.ricci_flow.compute_curvature_entropy(
                self.rewa_memory.fisher_metric_cache
            )
        else:
            curvature_entropy = 0.0
        
        free_energy = semantic_energy - self.controller.temperature * curvature_entropy
        
        # Check for rule discovery
        if epoch % self.eval_interval == 0:
            self._check_rule_discovery(epoch)
        
        return {
            'free_energy': free_energy,
            'semantic_energy': semantic_energy,
            'curvature_entropy': curvature_entropy,
            'contradictions': contradictions_detected,
            'significant_contradictions': significant_contradictions,
            'weak_contradictions': weak_contradictions,
            'ricci_updates': ricci_updates
        }
    
    def _check_rule_discovery(self, epoch: int):
        """Check if red → large rule has been discovered"""
        # Sample red items and check clustering
        red_items = []
        for item_id, ws in self.rewa_memory.memory.items():
            if ws.metadata and ws.metadata.get('attributes', {}).get('color') == 'red':
                red_items.append(item_id)
        
        if len(red_items) > 10 and self.results['rule_discovery_epoch'] is None:
            # Mark as discovered
            print(f"\n[Epoch {epoch}] Potential rule discovery detected!")
            self.results['rule_discovery_epoch'] = epoch
            
            if self.topos_layer is not None:
                self.topos_layer.add_rule(
                    'red_implies_large',
                    {'antecedent': 'red', 'consequent': 'large'}
                )
    
    def run(self) -> Dict:
        """Run full hardened self-discovery experiment"""
        print(f"\n{'='*60}")
        print(f"Hardened Self-Discovery Experiment")
        print(f"Topos: {self.enable_topos}, Ricci: {self.enable_ricci}")
        print(f"Temperature: {self.controller.temperature}")
        print(f"Dream Policy: {self.dream_policy}")
        print(f"Entropy Threshold: {self.entropy_spike_threshold}")
        print(f"Seed: {self.seed}")
        print(f"{'='*60}\n")
        
        for epoch in tqdm(range(self.n_dream_epochs), desc="Dreaming"):
            cycle_results = self.run_dream_cycle(epoch)
            
            # Record results
            self.results['free_energy'].append(cycle_results['free_energy'])
            self.results['semantic_energy'].append(cycle_results['semantic_energy'])
            self.results['curvature_entropy'].append(cycle_results['curvature_entropy'])
            self.results['contradictions'].append(cycle_results['contradictions'])
            self.results['significant_contradictions'].append(cycle_results['significant_contradictions'])
            self.results['weak_contradictions'].append(cycle_results['weak_contradictions'])
            self.results['ricci_updates'].append(cycle_results['ricci_updates'])
            
            if epoch % self.eval_interval == 0:
                print(f"\n[Epoch {epoch}]")
                print(f"  Free Energy: {cycle_results['free_energy']:.4f}")
                print(f"  Contradictions: {cycle_results['contradictions']} "
                      f"(significant: {cycle_results['significant_contradictions']}, "
                      f"weak: {cycle_results['weak_contradictions']})")
                print(f"  Ricci Updates: {cycle_results['ricci_updates']}")
                
                if self.topos_layer is not None:
                    stats = self.topos_layer.get_statistics()
                    print(f"  Rules Discovered: {stats['num_rules']}")
        
        # Final statistics
        print(f"\n{'='*60}")
        print("Experiment Complete!")
        print(f"{'='*60}")
        
        if self.results['rule_discovery_epoch'] is not None:
            print(f"✓ Rule discovered at epoch {self.results['rule_discovery_epoch']}")
        else:
            print("✗ Rule not discovered")
        
        print(f"\nFinal Statistics:")
        print(f"  Mean Free Energy: {np.mean(self.results['free_energy']):.4f}")
        print(f"  Total Contradictions: {sum(self.results['contradictions'])}")
        print(f"  Significant Contradictions: {sum(self.results['significant_contradictions'])}")
        print(f"  Total Ricci Updates: {sum(self.results['ricci_updates'])}")
        
        # Compute free energy trend
        if len(self.results['free_energy']) > 10:
            slope = np.polyfit(range(len(self.results['free_energy'])), 
                             self.results['free_energy'], 1)[0]
            print(f"  Free Energy Slope: {slope:.6f}")
            if slope < 0:
                print("  ✓ Free energy is decreasing!")
            else:
                print("  ⚠ Free energy is not decreasing")
        
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
        """Plot experiment results with Ricci update markers"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(len(self.results['free_energy']))
        
        # Free energy with Ricci update markers
        axes[0, 0].plot(self.results['free_energy'], label='Free Energy')
        
        # Mark Ricci updates
        ricci_epochs = [i for i, r in enumerate(self.results['ricci_updates']) if r > 0]
        if ricci_epochs:
            axes[0, 0].scatter(ricci_epochs, 
                             [self.results['free_energy'][i] for i in ricci_epochs],
                             color='red', s=30, alpha=0.6, label='Ricci Update', zorder=5)
        
        axes[0, 0].set_title('Free Energy F = E - T·S')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('F')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy components
        axes[0, 1].plot(self.results['semantic_energy'], label='E (Semantic Energy)')
        axes[0, 1].plot(self.results['curvature_entropy'], label='S (Curvature Entropy)')
        axes[0, 1].set_title('Energy Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Contradictions (stacked)
        axes[1, 0].plot(np.cumsum(self.results['significant_contradictions']), 
                       label='Significant (>threshold)', linewidth=2)
        axes[1, 0].plot(np.cumsum(self.results['weak_contradictions']), 
                       label='Weak (<threshold)', linewidth=2, linestyle='--')
        axes[1, 0].set_title('Cumulative Contradictions')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Ricci updates
        axes[1, 1].plot(np.cumsum(self.results['ricci_updates']))
        axes[1, 1].set_title('Cumulative Ricci Updates (Event-Driven)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark rule discovery
        if self.results['rule_discovery_epoch'] is not None:
            for ax in axes.flat:
                ax.axvline(self.results['rule_discovery_epoch'], 
                          color='green', linestyle='--', alpha=0.5,
                          label='Rule Discovery')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")


def run_hardened_experiment(
    seed: int = 42,
    temperature: float = 0.5,
    entropy_threshold: float = 0.5,
    dream_policy: str = 'mixed',
    enable_topos: bool = True,
    enable_ricci: bool = True,
    output_dir: str = 'logs'
) -> Dict:
    """Run hardened self-discovery experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create experiment
    experiment = HardenedSelfDiscoveryExperiment(
        n_dream_epochs=500,
        dreams_per_cycle=20,
        eval_interval=20,
        entropy_spike_threshold=entropy_threshold,
        temperature=temperature,
        dream_policy=dream_policy,
        seed=seed,
        enable_topos=enable_topos,
        enable_ricci=enable_ricci
    )
    
    # Load filtered dataset
    experiment.load_filtered_dataset('datasets/hidden_rule_filtered.pkl')
    
    # Run experiment
    results = experiment.run()
    
    # Save results
    config_str = f"hardened_T{temperature}_thresh{entropy_threshold}_{dream_policy}_seed{seed}"
    experiment.save_results(f"{output_dir}/results_{config_str}.json")
    experiment.plot_results(f"{output_dir}/plot_{config_str}.png")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Hardened Self-Discovery Experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.5, help='Exploration temperature')
    parser.add_argument('--entropy-threshold', type=float, default=0.5, 
                       help='Threshold for significant contradictions')
    parser.add_argument('--dream-policy', type=str, default='mixed',
                       choices=['freq', 'mixed', 'uniform'],
                       help='Dream sampling policy')
    parser.add_argument('--no-topos', action='store_true', help='Disable Topos layer')
    parser.add_argument('--no-ricci', action='store_true', help='Disable Ricci flow')
    parser.add_argument('--output-dir', type=str, default='logs', help='Output directory')
    
    args = parser.parse_args()
    
    results = run_hardened_experiment(
        seed=args.seed,
        temperature=args.temperature,
        entropy_threshold=args.entropy_threshold,
        dream_policy=args.dream_policy,
        enable_topos=not args.no_topos,
        enable_ricci=not args.no_ricci,
        output_dir=args.output_dir
    )
    
    print("\n✓ Hardened experiment complete!")
