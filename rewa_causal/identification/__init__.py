"""
Identification Layer

Implements causal identification methods:
- Backdoor adjustment
- Front-door adjustment
- Interventional region construction
- Causal effect computation
"""

from .adjustment import (
    backdoor_adjustment,
    frontdoor_adjustment,
    interventional_region,
    causal_effect
)

__all__ = [
    'backdoor_adjustment',
    'frontdoor_adjustment',
    'interventional_region',
    'causal_effect'
]
