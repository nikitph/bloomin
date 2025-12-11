"""
Refusal & Safety Layer

Ensures causal claims are only made when geometry is sound.
Never outputs a causal claim unless all validation passes.
"""

from .validation import (
    RefusalResult,
    validate_region,
    validate_causal_claim,
    ValidationConfig,
    CausalClaimValidator
)

__all__ = [
    'RefusalResult',
    'validate_region',
    'validate_causal_claim',
    'ValidationConfig',
    'CausalClaimValidator'
]
