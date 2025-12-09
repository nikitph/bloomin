import logging
from typing import List, Dict, Any

from topos.reasoning import ReasoningLayer, ReasoningResult
# Use relative import for models to avoid circular dependencies if any
from .models import VerificationResult, ChunkVerification

logger = logging.getLogger(__name__)

class ReasoningFilter:
    """
    Stateless middleware that uses `ReasoningLayer` to sanitize RAG context.
    """
    
    def __init__(self):
        self.layer = ReasoningLayer()
        
    def verify_context(self, query: str, chunks: List[str]) -> VerificationResult:
        """
        Verify that retrieved chunks are consistent with each other and the query.
        
        Args:
            query: User's natural language query.
            chunks: List of text chunks retrieved from Vector DB.
            
        Returns:
            VerificationResult with detailed per-chunk status.
        """
        # 1. Check Query Feasibility
        # Heuristic: Parse simple directional/topological constraints from query
        # In a real system, we'd use an NLP parser to extract these.
        # For this MVP, we look for keywords manually.
        query_constraints = []
        if "north of north pole" in query.lower():
            query_constraints.append({"relation": "North", "target": "North Pole"})
            
        feasibility = self.layer.check_query_feasibility(query_constraints)
        
        if feasibility.status == "impossible":
            return VerificationResult(
                query=query,
                verified_chunks=[],
                overall_confidence=0.0,
                global_warnings=[f"IMPOSSIBLE QUERY: {feasibility.explanation}"]
            )
            
        # 2. Verify Chunks (Modifier Binding & Consistency)
        verified_chunks = []
        global_warnings = []
        
        for i, chunk in enumerate(chunks):
            # Check for Bad Modifiers in the chunk relative to query terms
            # Example: Query "Red Bike", Chunk mentions "Blue Bike" -> Potential contradiction if strict
            # Example: Query "Real Gun", Chunk mentions "Fake Gun" -> Hard modifier conflict
            
            chunk_ver = ChunkVerification(
                text=chunk,
                is_consistent=True,
                consistency_score=1.0,
                modifier_score=1.0,
                flags=[],
                explanation=""
            )
            
            chunk_lower = chunk.lower()
            
            # --- Hard Modifier Check using ReasoningLayer ---
            # We scan for known hard modifiers present in the chunk
            modifiers_to_check = ["fake", "toy", "replica", "stone", "statue"]
            for mod in modifiers_to_check:
                if mod in chunk_lower:
                    # Let's see what the layer thinks about this modifier
                    # We assume the noun is generic "Object" or try to find one
                    # This is a heuristic for the MVP.
                    # If query asks for "Real" or implied real, and we find "Fake", flag it.
                    
                    # Heuristic: If query does NOT contain the modifier, but chunk DOES, 
                    # and the modifier is 'property-negating', we flag it.
                    if mod not in query.lower():
                        # Test the modifier impact
                        _, props = self.layer.apply_modifier("Object", mod.capitalize(), {"dangerous": 1.0})
                        # Check if dangerous is negated (0.0) or if it became explicitly artificial
                        if props.get("dangerous") == 0.0 or props.get("is_artificial") == 1.0:
                            chunk_ver.modifier_score = 0.5
                            chunk_ver.flags.append("HARD_MODIFIER_DETECTED")
                            chunk_ver.explanation += f"Contains negation modifier '{mod}' not requested in query. "
            
            # --- Simple Consistency Check ---
            # If the chunk explicitly negates a query term
            # Query: "Red", Chunk: "Not Red" (very naive check)
            
            if "not " in chunk_lower:
                # Naive: just flag potential negation for review
                # In full version, ToposLogic would build Sections and glue.
                chunk_ver.consistency_score = 0.9
                chunk_ver.flags.append("NEGATION_PRESENT")
                
            verified_chunks.append(chunk_ver)

        # 3. Calculate Overall Confidence
        # Average of chunk scores
        if not verified_chunks:
            avg_score = 0.0
        else:
            avg_score = sum(c.consistency_score * c.modifier_score for c in verified_chunks) / len(verified_chunks)

        return VerificationResult(
            query=query,
            verified_chunks=verified_chunks,
            overall_confidence=avg_score,
            global_warnings=global_warnings
        )
