"""Topos logic package"""

from .logic import (
    Proposition,
    LocalSection,
    ToposLogic,
    CompositionalQA
)
from .reasoning import ReasoningLayer, ReasoningResult
from .real_reasoning import RealReasoningLayer, ModifierResult
from .section_extractor import SectionExtractor, ExtractedEntity, extract_and_query

__all__ = [
    'Proposition',
    'LocalSection',
    'ToposLogic',
    'CompositionalQA',
    'ReasoningLayer',
    'ReasoningResult',
    'RealReasoningLayer',
    'ModifierResult',
    'SectionExtractor',
    'ExtractedEntity',
    'extract_and_query'
]
