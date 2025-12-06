"""
Topos Layer Module

Implements open set construction, gluing operations, consistency checking,
and contradiction detection for topological reasoning.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from scipy.stats import entropy as kl_divergence
from scipy.special import rel_entr


@dataclass
class OpenSet:
    """Topological open set with empirical distribution"""
    prototype_id: str
    item_ids: Set[str]
    radius: float
    distribution: np.ndarray  # Empirical probability distribution over witnesses
    metadata: Dict


@dataclass
class ConstraintSpec:
    """Specification of a detected contradiction"""
    region_ids: List[str]
    expected_distribution: np.ndarray
    observed_distribution: np.ndarray
    kl_divergence: float
    contradiction_type: str  # 'gluing_failure', 'null_space', 'inconsistency'


class ToposLayer:
    """
    Topological reasoning layer using sheaf-theoretic gluing.
    
    Implements:
    - Open set construction around prototypes
    - Gluing consistency checks via KL divergence
    - Null-space detection for impossible queries
    - Contradiction extraction for Ricci flow
    """
    
    def __init__(
        self,
        kl_threshold: float = 0.1,
        min_support: int = 3,
        null_space_threshold: float = 0.01
    ):
        self.kl_threshold = kl_threshold
        self.min_support = min_support
        self.null_space_threshold = null_space_threshold
        
        # Storage for discovered rules and constraints
        self.rules: Dict[str, Dict] = {}
        self.open_sets: Dict[str, OpenSet] = {}
    
    def build_open_set(
        self,
        prototype_id: str,
        prototype_witnesses: np.ndarray,
        memory_items: Dict,
        radius: float = 0.3
    ) -> OpenSet:
        """
        Construct an open set around a prototype.
        
        Args:
            prototype_id: Identifier for the prototype
            prototype_witnesses: Witness representation of prototype
            memory_items: Dictionary of item_id -> WitnessSet
            radius: Hamming radius for inclusion
            
        Returns:
            OpenSet containing items within radius
        """
        included_items = set()
        witness_vectors = []
        
        for item_id, witness_set in memory_items.items():
            # Compute Hamming distance
            hamming_dist = np.mean(prototype_witnesses != witness_set.witnesses)
            
            if hamming_dist <= radius:
                included_items.add(item_id)
                # Flatten witnesses for distribution
                witness_vectors.append(witness_set.witnesses.flatten())
        
        # Compute empirical distribution over witness patterns
        if len(witness_vectors) > 0:
            witness_vectors = np.array(witness_vectors)
            # Distribution = histogram of witness activations
            distribution = np.mean(witness_vectors, axis=0)
        else:
            # Empty set - uniform distribution
            distribution = np.ones(prototype_witnesses.size) * 0.5
        
        open_set = OpenSet(
            prototype_id=prototype_id,
            item_ids=included_items,
            radius=radius,
            distribution=distribution,
            metadata={'size': len(included_items)}
        )
        
        self.open_sets[prototype_id] = open_set
        return open_set
    
    def glue(
        self,
        open_sets: List[OpenSet],
        return_kl_matrix: bool = True
    ) -> Tuple[Set[str], Optional[np.ndarray], bool]:
        """
        Attempt to glue open sets and check consistency.
        
        Args:
            open_sets: List of open sets to glue
            return_kl_matrix: Whether to return pairwise KL matrix
            
        Returns:
            (glued_item_set, kl_matrix, is_consistent)
        """
        if len(open_sets) == 0:
            return set(), None, True
        
        if len(open_sets) == 1:
            return open_sets[0].item_ids, None, True
        
        # Find intersection (items in all open sets)
        glued_set = open_sets[0].item_ids.copy()
        for open_set in open_sets[1:]:
            glued_set = glued_set.intersection(open_set.item_ids)
        
        # Compute pairwise KL divergences
        n_sets = len(open_sets)
        kl_matrix = np.zeros((n_sets, n_sets))
        
        for i in range(n_sets):
            for j in range(i + 1, n_sets):
                # Symmetrized KL divergence
                p = open_sets[i].distribution + 1e-10  # Regularize
                q = open_sets[j].distribution + 1e-10
                
                # Normalize
                p = p / np.sum(p)
                q = q / np.sum(q)
                
                kl_pq = np.sum(rel_entr(p, q))
                kl_qp = np.sum(rel_entr(q, p))
                kl_sym = (kl_pq + kl_qp) / 2.0
                
                kl_matrix[i, j] = kl_sym
                kl_matrix[j, i] = kl_sym
        
        # Check consistency: all pairwise KL below threshold
        max_kl = np.max(kl_matrix)
        is_consistent = max_kl < self.kl_threshold
        
        if return_kl_matrix:
            return glued_set, kl_matrix, is_consistent
        else:
            return glued_set, None, is_consistent
    
    def detect_null_space(self, open_sets: List[OpenSet]) -> bool:
        """
        Detect if query corresponds to null space (impossible concept).
        
        Args:
            open_sets: Open sets from query
            
        Returns:
            True if null space detected
        """
        if len(open_sets) == 0:
            return True
        
        # Glue and check if intersection is empty or too small
        glued_set, _, _ = self.glue(open_sets, return_kl_matrix=False)
        
        if len(glued_set) < self.min_support:
            return True
        
        # Check if distributions are too inconsistent (high KL)
        _, kl_matrix, is_consistent = self.glue(open_sets, return_kl_matrix=True)
        
        if not is_consistent:
            # High inconsistency might indicate impossible combination
            max_kl = np.max(kl_matrix) if kl_matrix is not None else 0.0
            if max_kl > 1.0:  # Very high KL suggests null space
                return True
        
        return False
    
    def get_contradiction(self, open_sets: List[OpenSet]) -> Optional[ConstraintSpec]:
        """
        Extract contradiction specification from inconsistent gluing.
        
        Args:
            open_sets: Open sets that failed to glue consistently
            
        Returns:
            ConstraintSpec describing the contradiction, or None if consistent
        """
        if len(open_sets) < 2:
            return None
        
        glued_set, kl_matrix, is_consistent = self.glue(open_sets)
        
        if is_consistent:
            return None
        
        # Find pair with maximum KL divergence
        max_kl = 0.0
        max_i, max_j = 0, 1
        
        for i in range(len(open_sets)):
            for j in range(i + 1, len(open_sets)):
                if kl_matrix[i, j] > max_kl:
                    max_kl = kl_matrix[i, j]
                    max_i, max_j = i, j
        
        # Create constraint spec
        constraint = ConstraintSpec(
            region_ids=[open_sets[max_i].prototype_id, open_sets[max_j].prototype_id],
            expected_distribution=open_sets[max_i].distribution,
            observed_distribution=open_sets[max_j].distribution,
            kl_divergence=max_kl,
            contradiction_type='gluing_failure'
        )
        
        return constraint
    
    def add_rule(self, rule_name: str, rule_spec: Dict):
        """
        Add a discovered rule to the rule store.
        
        Args:
            rule_name: Name/identifier for the rule
            rule_spec: Dictionary specifying the rule (e.g., {'antecedent': 'red', 'consequent': 'large'})
        """
        self.rules[rule_name] = rule_spec
    
    def has_rule(self, rule_name: str) -> bool:
        """Check if a rule has been discovered"""
        return rule_name in self.rules
    
    def check_rule_consistency(
        self,
        rule_name: str,
        test_cases: List[Tuple[str, str, bool]]
    ) -> float:
        """
        Check consistency of a rule against test cases.
        
        Args:
            rule_name: Rule to check
            test_cases: List of (antecedent_id, consequent_id, expected_result)
            
        Returns:
            Accuracy on test cases
        """
        if rule_name not in self.rules:
            return 0.0
        
        rule = self.rules[rule_name]
        correct = 0
        
        for antecedent_id, consequent_id, expected in test_cases:
            # Simple rule checking (would be more sophisticated in practice)
            predicted = (
                antecedent_id == rule.get('antecedent') and
                consequent_id == rule.get('consequent')
            )
            if predicted == expected:
                correct += 1
        
        return correct / len(test_cases) if test_cases else 0.0
    
    def get_statistics(self) -> Dict:
        """Get Topos layer statistics"""
        return {
            'num_rules': len(self.rules),
            'num_open_sets': len(self.open_sets),
            'rules': list(self.rules.keys())
        }
