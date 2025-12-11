"""
Synthetic Data Generators for Testing REWA-Causal

Generates datasets with known causal structures for validation.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

try:
    from ..witness import WitnessSet
    from ..causal_graph import CausalGraph
except ImportError:
    from witness import WitnessSet
    from causal_graph import CausalGraph


@dataclass
class SyntheticDataset:
    """A synthetic dataset with known causal structure."""
    witnesses: Dict[str, WitnessSet]
    graph: CausalGraph
    ground_truth: Dict[str, Any]
    description: str


def generate_confounded_dataset(
    n_samples: int = 100,
    dimension: int = 64,
    confounder_strength: float = 0.5,
    treatment_effect: float = 0.3,
    seed: Optional[int] = 42
) -> SyntheticDataset:
    """
    Generate a dataset with confounding: U → X, U → Y, X → Y

    This is a classic case requiring backdoor adjustment.

    Args:
        n_samples: Number of samples per variable
        dimension: Embedding dimension
        confounder_strength: How strongly U affects X and Y
        treatment_effect: Direct effect of X on Y
        seed: Random seed

    Returns:
        SyntheticDataset with confounded structure
    """
    np.random.seed(seed)

    # Create graph
    graph = CausalGraph(
        nodes=['U', 'X', 'Y'],
        edges=[('U', 'X'), ('U', 'Y'), ('X', 'Y')]
    )
    graph._nodes['U'].is_observed = False
    graph._nodes['U'].is_latent = True

    # Generate latent confounder U
    # U lives on a specific region of the sphere
    U_center = np.random.randn(dimension)
    U_center = U_center / np.linalg.norm(U_center)

    U_witnesses = []
    for _ in range(n_samples):
        # Sample around U_center
        noise = np.random.randn(dimension) * 0.2
        u = U_center + noise
        u = u / np.linalg.norm(u)
        U_witnesses.append(u)
    U_witnesses = np.array(U_witnesses)

    # Generate X influenced by U
    X_base = np.random.randn(dimension)
    X_base = X_base / np.linalg.norm(X_base)

    X_witnesses = []
    for i in range(n_samples):
        # X = blend of X_base and U (confounder influence)
        x = (1 - confounder_strength) * X_base + confounder_strength * U_witnesses[i]
        noise = np.random.randn(dimension) * 0.1
        x = x + noise
        x = x / np.linalg.norm(x)
        X_witnesses.append(x)
    X_witnesses = np.array(X_witnesses)

    # Generate Y influenced by both U and X
    Y_base = np.random.randn(dimension)
    Y_base = Y_base / np.linalg.norm(Y_base)

    Y_witnesses = []
    for i in range(n_samples):
        # Y = blend of Y_base, U (confounder), and X (treatment)
        y = (1 - confounder_strength - treatment_effect) * Y_base
        y = y + confounder_strength * U_witnesses[i]
        y = y + treatment_effect * X_witnesses[i]
        noise = np.random.randn(dimension) * 0.1
        y = y + noise
        y = y / np.linalg.norm(y)
        Y_witnesses.append(y)
    Y_witnesses = np.array(Y_witnesses)

    witnesses = {
        'U': WitnessSet(variable='U', witnesses=U_witnesses),
        'X': WitnessSet(variable='X', witnesses=X_witnesses),
        'Y': WitnessSet(variable='Y', witnesses=Y_witnesses)
    }

    ground_truth = {
        'true_effect': treatment_effect,
        'confounder_strength': confounder_strength,
        'requires_adjustment': True,
        'valid_adjustment_sets': [{'U'}],  # If U were observed
        'X_center': X_base.tolist(),
        'Y_center': Y_base.tolist()
    }

    return SyntheticDataset(
        witnesses=witnesses,
        graph=graph,
        ground_truth=ground_truth,
        description="Confounded dataset: U → X, U → Y, X → Y"
    )


def generate_mediated_dataset(
    n_samples: int = 100,
    dimension: int = 64,
    x_to_m_effect: float = 0.4,
    m_to_y_effect: float = 0.4,
    seed: Optional[int] = 42
) -> SyntheticDataset:
    """
    Generate a dataset with mediation: X → M → Y

    No confounding, so no adjustment needed.

    Args:
        n_samples: Number of samples
        dimension: Embedding dimension
        x_to_m_effect: Effect of X on M
        m_to_y_effect: Effect of M on Y
        seed: Random seed

    Returns:
        SyntheticDataset with mediation structure
    """
    np.random.seed(seed)

    graph = CausalGraph(
        nodes=['X', 'M', 'Y'],
        edges=[('X', 'M'), ('M', 'Y')]
    )

    # Generate X
    X_center = np.random.randn(dimension)
    X_center = X_center / np.linalg.norm(X_center)

    X_witnesses = []
    for _ in range(n_samples):
        noise = np.random.randn(dimension) * 0.2
        x = X_center + noise
        x = x / np.linalg.norm(x)
        X_witnesses.append(x)
    X_witnesses = np.array(X_witnesses)

    # Generate M influenced by X
    M_base = np.random.randn(dimension)
    M_base = M_base / np.linalg.norm(M_base)

    M_witnesses = []
    for i in range(n_samples):
        m = (1 - x_to_m_effect) * M_base + x_to_m_effect * X_witnesses[i]
        noise = np.random.randn(dimension) * 0.1
        m = m + noise
        m = m / np.linalg.norm(m)
        M_witnesses.append(m)
    M_witnesses = np.array(M_witnesses)

    # Generate Y influenced by M
    Y_base = np.random.randn(dimension)
    Y_base = Y_base / np.linalg.norm(Y_base)

    Y_witnesses = []
    for i in range(n_samples):
        y = (1 - m_to_y_effect) * Y_base + m_to_y_effect * M_witnesses[i]
        noise = np.random.randn(dimension) * 0.1
        y = y + noise
        y = y / np.linalg.norm(y)
        Y_witnesses.append(y)
    Y_witnesses = np.array(Y_witnesses)

    witnesses = {
        'X': WitnessSet(variable='X', witnesses=X_witnesses),
        'M': WitnessSet(variable='M', witnesses=M_witnesses),
        'Y': WitnessSet(variable='Y', witnesses=Y_witnesses)
    }

    # Total effect = X→M effect * M→Y effect
    total_effect = x_to_m_effect * m_to_y_effect

    ground_truth = {
        'x_to_m_effect': x_to_m_effect,
        'm_to_y_effect': m_to_y_effect,
        'total_effect': total_effect,
        'requires_adjustment': False
    }

    return SyntheticDataset(
        witnesses=witnesses,
        graph=graph,
        ground_truth=ground_truth,
        description="Mediated dataset: X → M → Y"
    )


def generate_frontdoor_dataset(
    n_samples: int = 100,
    dimension: int = 64,
    confounder_strength: float = 0.4,
    x_to_m_effect: float = 0.5,
    m_to_y_effect: float = 0.5,
    seed: Optional[int] = 42
) -> SyntheticDataset:
    """
    Generate a dataset requiring front-door adjustment: X → M → Y, U → X, U → Y

    The mediator M blocks the confounding path.

    Args:
        n_samples: Number of samples
        dimension: Embedding dimension
        confounder_strength: Strength of confounding
        x_to_m_effect: Effect of X on M
        m_to_y_effect: Effect of M on Y
        seed: Random seed

    Returns:
        SyntheticDataset requiring front-door adjustment
    """
    np.random.seed(seed)

    graph = CausalGraph(
        nodes=['U', 'X', 'M', 'Y'],
        edges=[('U', 'X'), ('U', 'Y'), ('X', 'M'), ('M', 'Y')]
    )
    graph._nodes['U'].is_observed = False
    graph._nodes['U'].is_latent = True

    # Generate latent confounder U
    U_center = np.random.randn(dimension)
    U_center = U_center / np.linalg.norm(U_center)

    U_witnesses = []
    for _ in range(n_samples):
        noise = np.random.randn(dimension) * 0.2
        u = U_center + noise
        u = u / np.linalg.norm(u)
        U_witnesses.append(u)
    U_witnesses = np.array(U_witnesses)

    # Generate X influenced by U
    X_base = np.random.randn(dimension)
    X_base = X_base / np.linalg.norm(X_base)

    X_witnesses = []
    for i in range(n_samples):
        x = (1 - confounder_strength) * X_base + confounder_strength * U_witnesses[i]
        noise = np.random.randn(dimension) * 0.1
        x = x + noise
        x = x / np.linalg.norm(x)
        X_witnesses.append(x)
    X_witnesses = np.array(X_witnesses)

    # Generate M influenced ONLY by X (not by U - this is key for front-door)
    M_base = np.random.randn(dimension)
    M_base = M_base / np.linalg.norm(M_base)

    M_witnesses = []
    for i in range(n_samples):
        m = (1 - x_to_m_effect) * M_base + x_to_m_effect * X_witnesses[i]
        noise = np.random.randn(dimension) * 0.1
        m = m + noise
        m = m / np.linalg.norm(m)
        M_witnesses.append(m)
    M_witnesses = np.array(M_witnesses)

    # Generate Y influenced by M and U (but not directly by X)
    Y_base = np.random.randn(dimension)
    Y_base = Y_base / np.linalg.norm(Y_base)

    Y_witnesses = []
    for i in range(n_samples):
        y = (1 - m_to_y_effect - confounder_strength) * Y_base
        y = y + m_to_y_effect * M_witnesses[i]
        y = y + confounder_strength * U_witnesses[i]
        noise = np.random.randn(dimension) * 0.1
        y = y + noise
        y = y / np.linalg.norm(y)
        Y_witnesses.append(y)
    Y_witnesses = np.array(Y_witnesses)

    witnesses = {
        'U': WitnessSet(variable='U', witnesses=U_witnesses),
        'X': WitnessSet(variable='X', witnesses=X_witnesses),
        'M': WitnessSet(variable='M', witnesses=M_witnesses),
        'Y': WitnessSet(variable='Y', witnesses=Y_witnesses)
    }

    # True causal effect goes through M
    true_effect = x_to_m_effect * m_to_y_effect

    ground_truth = {
        'confounder_strength': confounder_strength,
        'x_to_m_effect': x_to_m_effect,
        'm_to_y_effect': m_to_y_effect,
        'true_causal_effect': true_effect,
        'requires_frontdoor': True,
        'mediator': 'M'
    }

    return SyntheticDataset(
        witnesses=witnesses,
        graph=graph,
        ground_truth=ground_truth,
        description="Front-door dataset: X → M → Y, U → X, U → Y"
    )


def generate_mortgage_causal_dataset(
    n_loans: int = 100,
    dimension: int = 64,
    seed: Optional[int] = 42
) -> SyntheticDataset:
    """
    Generate a mortgage-specific causal dataset.

    Graph: CreditScore → Default, DTI → Default, LTV → Default
           Economy (latent) → CreditScore, Economy → Default

    This models mortgage underwriting with latent economic confounding.

    Args:
        n_loans: Number of loan samples
        dimension: Embedding dimension
        seed: Random seed

    Returns:
        SyntheticDataset for mortgage causal inference
    """
    np.random.seed(seed)

    graph = CausalGraph(
        nodes=['Economy', 'CreditScore', 'DTI', 'LTV', 'Default'],
        edges=[
            ('Economy', 'CreditScore'),
            ('Economy', 'Default'),
            ('CreditScore', 'Default'),
            ('DTI', 'Default'),
            ('LTV', 'Default')
        ]
    )
    graph._nodes['Economy'].is_observed = False
    graph._nodes['Economy'].is_latent = True

    # Economic conditions (latent)
    econ_center = np.random.randn(dimension)
    econ_center = econ_center / np.linalg.norm(econ_center)

    economy_witnesses = []
    for _ in range(n_loans):
        noise = np.random.randn(dimension) * 0.3
        e = econ_center + noise
        e = e / np.linalg.norm(e)
        economy_witnesses.append(e)
    economy_witnesses = np.array(economy_witnesses)

    # Credit Score (affected by economy)
    credit_base = np.random.randn(dimension)
    credit_base = credit_base / np.linalg.norm(credit_base)

    credit_witnesses = []
    for i in range(n_loans):
        c = 0.7 * credit_base + 0.3 * economy_witnesses[i]
        noise = np.random.randn(dimension) * 0.15
        c = c + noise
        c = c / np.linalg.norm(c)
        credit_witnesses.append(c)
    credit_witnesses = np.array(credit_witnesses)

    # DTI (independent)
    dti_base = np.random.randn(dimension)
    dti_base = dti_base / np.linalg.norm(dti_base)

    dti_witnesses = []
    for _ in range(n_loans):
        noise = np.random.randn(dimension) * 0.25
        d = dti_base + noise
        d = d / np.linalg.norm(d)
        dti_witnesses.append(d)
    dti_witnesses = np.array(dti_witnesses)

    # LTV (independent)
    ltv_base = np.random.randn(dimension)
    ltv_base = ltv_base / np.linalg.norm(ltv_base)

    ltv_witnesses = []
    for _ in range(n_loans):
        noise = np.random.randn(dimension) * 0.25
        l = ltv_base + noise
        l = l / np.linalg.norm(l)
        ltv_witnesses.append(l)
    ltv_witnesses = np.array(ltv_witnesses)

    # Default (affected by all factors)
    default_base = np.random.randn(dimension)
    default_base = default_base / np.linalg.norm(default_base)

    default_witnesses = []
    for i in range(n_loans):
        # Negative credit effect (higher credit = lower default)
        d = 0.3 * default_base
        d = d - 0.25 * credit_witnesses[i]  # Negative: good credit reduces default
        d = d + 0.2 * dti_witnesses[i]      # Positive: high DTI increases default
        d = d + 0.15 * ltv_witnesses[i]     # Positive: high LTV increases default
        d = d + 0.2 * economy_witnesses[i]  # Economic conditions affect default

        noise = np.random.randn(dimension) * 0.1
        d = d + noise
        d = d / np.linalg.norm(d)
        default_witnesses.append(d)
    default_witnesses = np.array(default_witnesses)

    witnesses = {
        'Economy': WitnessSet(variable='Economy', witnesses=economy_witnesses),
        'CreditScore': WitnessSet(variable='CreditScore', witnesses=credit_witnesses),
        'DTI': WitnessSet(variable='DTI', witnesses=dti_witnesses),
        'LTV': WitnessSet(variable='LTV', witnesses=ltv_witnesses),
        'Default': WitnessSet(variable='Default', witnesses=default_witnesses)
    }

    ground_truth = {
        'credit_effect': -0.25,
        'dti_effect': 0.2,
        'ltv_effect': 0.15,
        'economy_confounding': 0.2,
        'valid_adjustment_for_credit': {'DTI', 'LTV'},  # Can adjust for DTI/LTV but not Economy
        'confounded_variables': ['CreditScore']
    }

    return SyntheticDataset(
        witnesses=witnesses,
        graph=graph,
        ground_truth=ground_truth,
        description="Mortgage causal dataset with economic confounding"
    )
