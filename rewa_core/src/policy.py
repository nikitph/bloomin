"""
Policy Ingestion (Bias Compiler)

Compiles natural-language policies, historical outcomes, and risk postures
into prior functions ρ(μ) that score candidate meanings.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import json
import hashlib


class RiskPosture(Enum):
    """Risk posture selection for policy behavior."""
    CONSERVATIVE = "conservative"  # Prefer refusal over incorrect approval
    MODERATE = "moderate"          # Balanced approach
    PERMISSIVE = "permissive"      # Prefer approval, minimize refusals


@dataclass
class PolicySpec:
    """Specification for a policy."""
    id: str
    name: str
    description: str
    risk_posture: RiskPosture = RiskPosture.MODERATE
    rules: List[str] = field(default_factory=list)  # Natural language rules
    prototypes: List[str] = field(default_factory=list)  # Positive examples
    antiprototypes: List[str] = field(default_factory=list)  # Negative examples
    threshold: float = 0.5  # Minimum score for approval
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.id:
            # Generate ID from name
            self.id = hashlib.sha256(self.name.encode()).hexdigest()[:16]


class PolicyFunction(ABC):
    """Abstract base class for policy scoring functions."""

    @abstractmethod
    def score(self, meaning: np.ndarray) -> float:
        """Score a candidate meaning. Higher = more preferred."""
        pass

    @abstractmethod
    def batch_score(self, meanings: np.ndarray) -> np.ndarray:
        """Score multiple meanings efficiently."""
        pass


class PrototypeSimilarityPolicy(PolicyFunction):
    """
    Policy based on similarity to prototype embeddings.

    Score = max similarity to prototypes - max similarity to antiprototypes
    """

    def __init__(
        self,
        prototypes: np.ndarray,
        antiprototypes: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ):
        self.prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
        self.antiprototypes = None
        if antiprototypes is not None and len(antiprototypes) > 0:
            self.antiprototypes = antiprototypes / np.linalg.norm(
                antiprototypes, axis=1, keepdims=True
            )
        self.weights = weights

    def score(self, meaning: np.ndarray) -> float:
        meaning = meaning / (np.linalg.norm(meaning) + 1e-10)

        # Similarity to prototypes
        proto_sims = self.prototypes @ meaning
        if self.weights is not None:
            proto_score = float(np.sum(self.weights * proto_sims))
        else:
            proto_score = float(np.max(proto_sims))

        # Penalty from antiprototypes
        anti_score = 0.0
        if self.antiprototypes is not None:
            anti_sims = self.antiprototypes @ meaning
            anti_score = float(np.max(anti_sims))

        return proto_score - anti_score

    def batch_score(self, meanings: np.ndarray) -> np.ndarray:
        meanings = meanings / (np.linalg.norm(meanings, axis=1, keepdims=True) + 1e-10)

        # Prototype similarities
        proto_sims = meanings @ self.prototypes.T  # (n_meanings, n_prototypes)
        if self.weights is not None:
            proto_scores = proto_sims @ self.weights
        else:
            proto_scores = np.max(proto_sims, axis=1)

        # Anti-prototype penalties
        anti_scores = np.zeros(len(meanings))
        if self.antiprototypes is not None:
            anti_sims = meanings @ self.antiprototypes.T
            anti_scores = np.max(anti_sims, axis=1)

        return proto_scores - anti_scores


class MLPScoringPolicy(PolicyFunction):
    """
    Policy using a small MLP scoring head.

    Trained on historical labeled outcomes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        weights: Optional[Dict[str, np.ndarray]] = None
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Initialize or load weights
        if weights:
            self.weights = weights
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize MLP weights."""
        self.weights = {}
        prev_dim = self.input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            self.weights[f'W{i}'] = np.random.randn(prev_dim, hidden_dim) * 0.1
            self.weights[f'b{i}'] = np.zeros(hidden_dim)
            prev_dim = hidden_dim

        # Output layer (score)
        self.weights['W_out'] = np.random.randn(prev_dim, 1) * 0.1
        self.weights['b_out'] = np.zeros(1)

    def score(self, meaning: np.ndarray) -> float:
        return float(self.batch_score(meaning.reshape(1, -1))[0])

    def batch_score(self, meanings: np.ndarray) -> np.ndarray:
        x = meanings

        # Forward through hidden layers
        for i in range(len(self.hidden_dims)):
            x = x @ self.weights[f'W{i}'] + self.weights[f'b{i}']
            x = np.maximum(0, x)  # ReLU

        # Output layer
        x = x @ self.weights['W_out'] + self.weights['b_out']
        return x.flatten()

    def train(
        self,
        meanings: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100
    ):
        """Train the MLP on labeled examples."""
        # Simple gradient descent (production would use PyTorch)
        for epoch in range(epochs):
            # Forward pass
            scores = self.batch_score(meanings)

            # MSE loss
            loss = np.mean((scores - labels) ** 2)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

            # Backward pass (simplified)
            # This is a placeholder - production would use autograd
            pass


class HybridPolicy(PolicyFunction):
    """
    Hybrid policy combining prototype similarity and MLP scoring.

    score = α * prototype_score + (1 - α) * mlp_score
    """

    def __init__(
        self,
        prototype_policy: PrototypeSimilarityPolicy,
        mlp_policy: MLPScoringPolicy,
        alpha: float = 0.5
    ):
        self.prototype_policy = prototype_policy
        self.mlp_policy = mlp_policy
        self.alpha = alpha

    def score(self, meaning: np.ndarray) -> float:
        proto_score = self.prototype_policy.score(meaning)
        mlp_score = self.mlp_policy.score(meaning)
        return self.alpha * proto_score + (1 - self.alpha) * mlp_score

    def batch_score(self, meanings: np.ndarray) -> np.ndarray:
        proto_scores = self.prototype_policy.batch_score(meanings)
        mlp_scores = self.mlp_policy.batch_score(meanings)
        return self.alpha * proto_scores + (1 - self.alpha) * mlp_scores


@dataclass
class Policy:
    """Complete policy with specification and scoring function."""
    spec: PolicySpec
    scoring_function: PolicyFunction
    is_active: bool = True

    def score(self, meaning: np.ndarray) -> float:
        """Score a candidate meaning."""
        return self.scoring_function.score(meaning)

    def batch_score(self, meanings: np.ndarray) -> np.ndarray:
        """Score multiple meanings."""
        return self.scoring_function.batch_score(meanings)

    def meets_threshold(self, meaning: np.ndarray) -> bool:
        """Check if meaning meets policy threshold."""
        return self.score(meaning) >= self.spec.threshold


class PolicyEngine:
    """
    Engine for compiling and managing policies.

    Handles:
    - Policy compilation from specs
    - Policy storage and retrieval
    - Multi-policy evaluation
    """

    def __init__(self, semantic_space):
        """
        Args:
            semantic_space: SemanticSpace instance for embedding policy text
        """
        self.semantic_space = semantic_space
        self.policies: Dict[str, Policy] = {}

    def compile(self, spec: PolicySpec) -> Policy:
        """
        Compile a PolicySpec into a usable Policy.

        Converts natural language rules and examples into scoring function.
        """
        # Embed prototypes
        prototypes = None
        if spec.prototypes:
            prototypes = self.semantic_space.embed_batch(spec.prototypes)

        # Embed antiprototypes
        antiprototypes = None
        if spec.antiprototypes:
            antiprototypes = self.semantic_space.embed_batch(spec.antiprototypes)

        # Embed rules (treat as additional prototypes)
        if spec.rules:
            rule_embeddings = self.semantic_space.embed_batch(spec.rules)
            if prototypes is not None:
                prototypes = np.vstack([prototypes, rule_embeddings])
            else:
                prototypes = rule_embeddings

        # Ensure we have at least some prototypes
        if prototypes is None or len(prototypes) == 0:
            # Default: prefer high-norm directions (arbitrary but deterministic)
            prototypes = np.eye(self.semantic_space.dimension)[:10]

        # Create scoring function based on risk posture
        if spec.risk_posture == RiskPosture.CONSERVATIVE:
            # Conservative: require high similarity, penalize antiprototypes more
            scoring_function = PrototypeSimilarityPolicy(
                prototypes, antiprototypes,
                weights=None  # Use max similarity
            )
        elif spec.risk_posture == RiskPosture.PERMISSIVE:
            # Permissive: accept average similarity
            weights = np.ones(len(prototypes)) / len(prototypes)
            scoring_function = PrototypeSimilarityPolicy(
                prototypes, antiprototypes,
                weights=weights
            )
        else:
            # Moderate: default max similarity
            scoring_function = PrototypeSimilarityPolicy(
                prototypes, antiprototypes
            )

        policy = Policy(
            spec=spec,
            scoring_function=scoring_function,
            is_active=True
        )

        # Store policy
        self.policies[spec.id] = policy

        return policy

    def get(self, policy_id: str) -> Optional[Policy]:
        """Get policy by ID."""
        return self.policies.get(policy_id)

    def list_active(self) -> List[Policy]:
        """List all active policies."""
        return [p for p in self.policies.values() if p.is_active]

    def evaluate_all(
        self,
        meaning: np.ndarray,
        policy_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate meaning against multiple policies.

        Returns:
            Dict mapping policy_id to {score, meets_threshold, risk_posture}
        """
        results = {}

        policies = [self.policies[pid] for pid in policy_ids] if policy_ids else self.list_active()

        for policy in policies:
            score = policy.score(meaning)
            results[policy.spec.id] = {
                'score': score,
                'meets_threshold': score >= policy.spec.threshold,
                'threshold': policy.spec.threshold,
                'risk_posture': policy.spec.risk_posture.value,
                'policy_name': policy.spec.name
            }

        return results

    def find_best_policy_meaning(
        self,
        candidates: np.ndarray,
        policy_id: str
    ) -> Tuple[np.ndarray, float, int]:
        """
        Find the candidate meaning that best satisfies a policy.

        Returns:
            (best_meaning, best_score, best_index)
        """
        policy = self.policies.get(policy_id)
        if policy is None:
            raise ValueError(f"Policy {policy_id} not found")

        scores = policy.batch_score(candidates)
        best_idx = int(np.argmax(scores))

        return candidates[best_idx], float(scores[best_idx]), best_idx

    def save_policy(self, policy_id: str, filepath: str):
        """Save policy specification to JSON."""
        policy = self.policies.get(policy_id)
        if policy is None:
            raise ValueError(f"Policy {policy_id} not found")

        spec_dict = {
            'id': policy.spec.id,
            'name': policy.spec.name,
            'description': policy.spec.description,
            'risk_posture': policy.spec.risk_posture.value,
            'rules': policy.spec.rules,
            'prototypes': policy.spec.prototypes,
            'antiprototypes': policy.spec.antiprototypes,
            'threshold': policy.spec.threshold,
            'metadata': policy.spec.metadata,
            'created_at': policy.spec.created_at
        }

        with open(filepath, 'w') as f:
            json.dump(spec_dict, f, indent=2)

    def load_policy(self, filepath: str) -> Policy:
        """Load and compile policy from JSON."""
        with open(filepath, 'r') as f:
            spec_dict = json.load(f)

        spec = PolicySpec(
            id=spec_dict['id'],
            name=spec_dict['name'],
            description=spec_dict['description'],
            risk_posture=RiskPosture(spec_dict['risk_posture']),
            rules=spec_dict.get('rules', []),
            prototypes=spec_dict.get('prototypes', []),
            antiprototypes=spec_dict.get('antiprototypes', []),
            threshold=spec_dict.get('threshold', 0.5),
            metadata=spec_dict.get('metadata', {}),
            created_at=spec_dict.get('created_at', datetime.now().isoformat())
        )

        return self.compile(spec)
