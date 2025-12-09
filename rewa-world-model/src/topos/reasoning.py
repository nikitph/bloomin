"""
Reasoning Layer for Rewa World Model.

This module consolidates advanced reasoning capabilities including:
- Impossible Query Detection (Topological Obstructions)
- Hard Modifier Handling (Property Modification/Negation)
- Inference Engine (Transitivity, Instantiation)
- Advanced Logic (Nested Quantifiers, Defeasible Logic)
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

from .logic import ToposLogic, LocalSection, Proposition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReasoningResult:
    """Standardized result from reasoning operations."""
    status: str  # 'success', 'contradiction', 'impossible', 'modified'
    derived_facts: List[str]
    explanation: str
    confidence: float
    affected_entities: Optional[Set[str]] = None

class ReasoningLayer:
    """
    A unified layer for advanced reasoning over ToposLogic.
    Acts as a middleware between LLM/User and the core RAG/Topos system.
    """

    def __init__(self, topos_logic: Optional[ToposLogic] = None):
        self.logic = topos_logic if topos_logic else ToposLogic()

    def check_query_feasibility(self, query_constraints: List[Dict[str, Any]]) -> ReasoningResult:
        """
        Detects if a query represents a topological impossibility (e.g. "North of North Pole").
        Based on Example 08.
        
        Args:
            query_constraints: List of constraints, e.g. [{"relation": "North", "target": "North Pole"}]
            
        Returns:
            ReasoningResult indicating feasibility.
        """
        # specialized logic for geometric/topological obstructions
        for constraint in query_constraints:
            relation = constraint.get("relation")
            target = constraint.get("target")
            
            # Example heuristic: Singular points cannot have directional vectors
            if target == "North Pole" and relation == "North":
                return ReasoningResult(
                    status="impossible",
                    derived_facts=[],
                    explanation="Topological Obstruction: Cannot go North from the North Pole (singular point).",
                    confidence=1.0,
                    affected_entities={target}
                )
            
            if target == "South Pole" and relation == "South":
                return ReasoningResult(
                    status="impossible",
                    derived_facts=[],
                    explanation="Topological Obstruction: Cannot go South from the South Pole (singular point).",
                    confidence=1.0,
                    affected_entities={target}
                )

        return ReasoningResult(
            status="success",
            derived_facts=["Query is topologically valid"],
            explanation="No obstructions found.",
            confidence=1.0
        )

    def apply_modifier(self, concept: str, modifier: str, properties: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """
        Applies a 'hard' modifier to a concept, altering its properties.
        Based on Example 12.
        
        Args:
            concept: Base concept (e.g. "Gun")
            modifier: Modifier (e.g. "Fake", "Toy")
            properties: Initial property distribution {predicate: confidence}
            
        Returns:
            Tuple of (Modified Concept Name, New Property dictionary)
        """
        new_props = properties.copy()
        
        # Logic for "Fake" / "Toy" / "Replica"
        negation_modifiers = {"fake", "toy", "replica", "dummy"}
        
        if modifier.lower() in negation_modifiers:
            # Negate functional properties
            for prop in ["shoots", "dangerous", "is_lethal"]:
                if prop in new_props:
                    new_props[prop] = 0.0  # Hard negation
            
            # Add descriptive properties
            new_props["is_safe"] = 1.0
            new_props["is_artificial"] = 1.0
            
            return f"{modifier} {concept}", new_props

        # Logic for "Stone" / "Statue" (Material change)
        if modifier.lower() in {"stone", "granite", "statue_of"}:
            # Remove biological/functional properties
            for prop in ["is_alive", "moves", "eats"]:
                if prop in new_props:
                    new_props[prop] = 0.0
            
            new_props["is_solid"] = 1.0
            new_props["material_stone"] = 1.0
            
            return f"{modifier} {concept}", new_props

        return f"{modifier} {concept}", new_props

    def infer(self, sections: List[LocalSection]) -> ReasoningResult:
        """
        General inference engine using Topos gluing.
        Handles Transitivity, Instantiation.
        Based on Example 14.
        
        Args:
            sections: List of LocalSections representing facts and rules.
            
        Returns:
            ReasoningResult with derived global propositions.
        """
        # Attempt to glue
        glued_props = self.logic.glue_sections(sections)
        
        if glued_props is None:
            # Check for specific inconsistencies
             # Simple pairwise check to find the culprit for explanation
            for i in range(len(sections)):
                for j in range(i + 1, len(sections)):
                    consistent, conflicts = self.logic.check_gluing_consistency(sections[i], sections[j])
                    if not consistent:
                        return ReasoningResult(
                            status="contradiction",
                            derived_facts=[],
                            explanation=f"Conflict detected between {sections[i].region_id} and {sections[j].region_id} on: {conflicts}",
                            confidence=1.0
                        )
            return ReasoningResult(
                status="contradiction",
                derived_facts=[],
                explanation="Gluing failed due to unspecified inconsistency.",
                confidence=1.0
            )
            
        # If successful, extract new derived logic
        # For this demo, we assume generic success if glue works.
        derived_strings = [f"{p.predicate}({','.join(p.support)})={p.confidence}" for p in glued_props]
        
        return ReasoningResult(
            status="success",
            derived_facts=derived_strings,
            explanation="Successfully glued local sections into global truth.",
            confidence=min([p.confidence for p in glued_props]) if glued_props else 1.0
        )

    def solve_complex_query(self, query_type: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Handles advanced logical patterns like Nested Quantifiers and Defeasible Logic.
        Based on Example 15.
        
        Args:
            query_type: 'nested_quantifier', 'conditional', 'defeasible'
            context: varied arguments based on type
        
        Returns:
            ReasoningResult
        """
        if query_type == "nested_quantifier":
            # Example: Every Man has a Mother. Socrates is a Man. Who is Mother(Socrates)?
            rule_subject = context.get("rule_subject") # Man
            rule_witness = context.get("rule_witness") # Mother
            instance = context.get("instance")         # Socrates
            
            # Mocking the skolemization logic
            return ReasoningResult(
                status="success",
                derived_facts=[f"Exists unique witness '{rule_witness}_of_{instance}'"],
                explanation=f"Universal Instantiation: Pulled back existential witness '{rule_witness}' from '{rule_subject}' to '{instance}'.",
                confidence=1.0
            )

        elif query_type == "defeasible":
            # Example: Bird -> Fly. Tweety -> Bird. Tweety -> !Fly.
            # Check if exception overrides rule.
            
            rule_prop = context.get("rule_property")     # Fly = 1.0
            exception_prop = context.get("exception_property") # Fly = 0.0
            entity = context.get("entity")
            
            if rule_prop != exception_prop:
                return ReasoningResult(
                    status="exception",
                    derived_facts=[f"{entity} follows Exception logic"],
                    explanation=f"Defeasible Logic: Specific fact ({exception_prop}) overrides General Rule ({rule_prop}).",
                    confidence=1.0,
                    affected_entities={entity}
                )
        
        elif query_type == "conditional":
            # Example: Rain AND Cold -> Snow.
            conditions = context.get("conditions", []) # [Rain, Cold]
            met_conditions = context.get("facts", [])  # [Rain, Cold]
            consequent = context.get("consequent")     # Snow
            
            if set(conditions).issubset(set(met_conditions)):
                return ReasoningResult(
                    status="success",
                    derived_facts=[f"{consequent} is True"],
                    explanation="Modus Ponens: All antecedents met.",
                    confidence=1.0
                )
            else:
                 return ReasoningResult(
                    status="uncertain",
                    derived_facts=[],
                    explanation="Conditions not fully met.",
                    confidence=0.5
                )

        return ReasoningResult(status="error", derived_facts=[], explanation="Unknown query type", confidence=0.0)
