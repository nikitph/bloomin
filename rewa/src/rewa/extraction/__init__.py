"""
Entity and Fact Extraction Module

Extracts structured knowledge from text chunks:
- Named entities with types
- Facts as subject-predicate-value triples
- Properties and attributes
- Temporal information
"""

from rewa.extraction.entity_extractor import EntityExtractor
from rewa.extraction.fact_extractor import FactExtractor
from rewa.extraction.query_compiler import QueryCompiler, compile_query

__all__ = [
    "EntityExtractor",
    "FactExtractor",
    "QueryCompiler",
    "compile_query",
]
