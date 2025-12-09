"""Witness extraction package"""

from .extractor import (
    Witness,
    WitnessType,
    WitnessExtractor,
    estimate_witness_distribution
)

__all__ = [
    'Witness',
    'WitnessType',
    'WitnessExtractor',
    'estimate_witness_distribution'
]
