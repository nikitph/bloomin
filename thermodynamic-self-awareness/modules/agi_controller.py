"""
AGI Controller Module

Orchestrates conscious cycles, manages hypotheses, and controls
the autopoietic loop.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class QuerySpec:
    """Specification for an internal query"""
    query_type: str  # 'dream', 'prediction', 'verification'
    concepts: List[str]
    expected_information_gain: float
    priority: float


@dataclass
class Diagnostics:
    """Diagnostics from a conscious cycle"""
    epoch: int
    free_energy: float
    semantic_energy: float
    curvature_entropy: float
    temperature: float
    contradictions_detected: int
    ricci_updates: int
    new_concepts: int
    rules_discovered: List[str]
    kl_divergences: List[float]
    timestamp: float


class AGIController:
    """
    Controller for autopoietic conscious cycles.
    
    Orchestrates:
    - Internal query planning based on information gain
    - Hypothesis generation and management
    - Conscious cycle execution
    - Free energy monitoring and optimization
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        query_budget: int = 20,
        eig_threshold: float = 0.1,
        seed: int = 42
    ):
        self.temperature = temperature
        self.query_budget = query_budget
        self.eig_threshold = eig_threshold
        self.rng = np.random.RandomState(seed)
        
        # State
        self.hypotheses: List[Dict] = []
        self.cycle_count: int = 0
        self.diagnostics_history: List[Diagnostics] = []
        
    def plan_internal_query(
        self,
        memory_stats: Dict,
        entropy_gradient: Optional[np.ndarray] = None
    ) -> QuerySpec:
        """
        Plan next internal query based on expected information gain.
        
        Args:
            memory_stats: Statistics from REWAMemory
            entropy_gradient: Optional gradient of entropy landscape
            
        Returns:
            QuerySpec for next query
        """
        # Simple heuristic: explore high-entropy regions
        if entropy_gradient is not None and np.linalg.norm(entropy_gradient) > 0:
            # Query in direction of entropy gradient
            query_type = 'dream'
            eig = np.linalg.norm(entropy_gradient)
        else:
            # Random exploration
            query_type = self.rng.choice(['dream', 'prediction', 'verification'])
            eig = self.rng.uniform(0, 1)
        
        # Select concepts to query (random for now)
        num_concepts = memory_stats.get('num_items', 0)
        if num_concepts > 0:
            n_query = min(3, num_concepts)
            concepts = [f"concept_{self.rng.randint(num_concepts)}" for _ in range(n_query)]
        else:
            concepts = []
        
        query = QuerySpec(
            query_type=query_type,
            concepts=concepts,
            expected_information_gain=eig,
            priority=eig
        )
        
        return query
    
    def manage_hypotheses(self, new_hypothesis: Optional[Dict] = None):
        """
        Add or prune hypotheses.
        
        Args:
            new_hypothesis: Optional new hypothesis to add
        """
        if new_hypothesis is not None:
            self.hypotheses.append(new_hypothesis)
        
        # Prune low-confidence hypotheses
        self.hypotheses = [h for h in self.hypotheses if h.get('confidence', 0) > 0.1]
    
    def conscious_cycle(
        self,
        rewa_memory,
        topos_layer,
        ricci_flow,
        semantic_rg,
        external_input: Optional[np.ndarray] = None
    ) -> Diagnostics:
        """
        Execute one conscious cycle of the autopoietic loop.
        
        Args:
            rewa_memory: REWAMemory instance
            topos_layer: ToposLayer instance
            ricci_flow: RicciFlow instance
            semantic_rg: SemanticRG instance
            external_input: Optional external input (if None, dream)
            
        Returns:
            Diagnostics from this cycle
        """
        start_time = time.time()
        
        # Step 1: Perception or Dreaming
        if external_input is not None:
            witnesses = rewa_memory.extract_witnesses(external_input)
            rewa_memory.store(witnesses)
            witness_list = [witnesses]
        else:
            # Dream: sample from manifold
            witness_list = rewa_memory.sample_from_manifold(n_samples=self.query_budget)
        
        # Step 2: Topos reasoning
        contradictions_detected = 0
        ricci_updates = 0
        kl_divergences = []
        
        if len(witness_list) > 1:
            # Build open sets
            open_sets = []
            for ws in witness_list:
                open_set = topos_layer.build_open_set(
                    prototype_id=ws.item_id,
                    prototype_witnesses=ws.witnesses,
                    memory_items=rewa_memory.memory,
                    radius=0.3
                )
                open_sets.append(open_set)
            
            # Attempt gluing
            glued_set, kl_matrix, is_consistent = topos_layer.glue(open_sets)
            
            if not is_consistent:
                contradictions_detected += 1
                
                # Get contradiction spec
                contradiction = topos_layer.get_contradiction(open_sets)
                
                if contradiction is not None:
                    kl_divergences.append(contradiction.kl_divergence)
                    
                    # Step 3: Ricci flow correction
                    current_metric = rewa_memory.fisher_metric(list(glued_set))
                    
                    error_signal = {
                        'kl_divergence': contradiction.kl_divergence,
                        'expected': contradiction.expected_distribution,
                        'observed': contradiction.observed_distribution
                    }
                    
                    delta_g = ricci_flow.flow_step(
                        current_metric=current_metric,
                        error_signal=error_signal,
                        region_ids=contradiction.region_ids
                    )
                    
                    # Apply update
                    new_metric = ricci_flow.apply_metric_update(current_metric, delta_g)
                    rewa_memory.fisher_metric_cache = new_metric
                    ricci_updates += 1
        
        # Step 4: Semantic RG consolidation
        new_concepts = 0
        if semantic_rg.should_consolidate(witness_list):
            packets = semantic_rg.coarse_grain(
                witness_sets=witness_list,
                current_scale=0,
                target_scale=1
            )
            
            for packet in packets:
                rewa_memory.store_abstraction(packet)
                new_concepts += 1
        
        # Step 5: Compute diagnostics
        semantic_energy = rewa_memory.compute_semantic_energy()
        
        if rewa_memory.fisher_metric_cache is not None:
            curvature_entropy = ricci_flow.compute_curvature_entropy(rewa_memory.fisher_metric_cache)
        else:
            curvature_entropy = 0.0
        
        free_energy = semantic_energy - self.temperature * curvature_entropy
        
        diagnostics = Diagnostics(
            epoch=self.cycle_count,
            free_energy=free_energy,
            semantic_energy=semantic_energy,
            curvature_entropy=curvature_entropy,
            temperature=self.temperature,
            contradictions_detected=contradictions_detected,
            ricci_updates=ricci_updates,
            new_concepts=new_concepts,
            rules_discovered=list(topos_layer.rules.keys()),
            kl_divergences=kl_divergences,
            timestamp=time.time() - start_time
        )
        
        self.diagnostics_history.append(diagnostics)
        self.cycle_count += 1
        
        return diagnostics
    
    def get_statistics(self) -> Dict:
        """Get controller statistics"""
        if len(self.diagnostics_history) == 0:
            return {
                'num_cycles': 0,
                'mean_free_energy': 0.0
            }
        
        return {
            'num_cycles': self.cycle_count,
            'num_hypotheses': len(self.hypotheses),
            'mean_free_energy': np.mean([d.free_energy for d in self.diagnostics_history]),
            'total_contradictions': sum(d.contradictions_detected for d in self.diagnostics_history),
            'total_ricci_updates': sum(d.ricci_updates for d in self.diagnostics_history)
        }
