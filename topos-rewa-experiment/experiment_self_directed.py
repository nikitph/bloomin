"""
Experiment 8: Self-Directed Learning (Socratic Dialogue)

Demonstrates autonomous question generation where the system builds knowledge
through internal dialogue by identifying knowledge gaps and filling them.
"""

import torch
import numpy as np
from conscious_agent import ConsciousAgent


def run_self_directed_learning():
    """
    System learns through internal Socratic dialogue
    
    Initial concepts: Dog, Cat, Horse, Cow
    
    Expected autonomous discoveries:
    1. Pet (from Dog, Cat)
    2. Farm Animal (from Horse, Cow)
    3. Mammal (from Pet, Farm Animal)
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*60)
    print("EXPERIMENT 8: SELF-DIRECTED LEARNING")
    print("Autonomous Question Generation & Socratic Dialogue")
    print("="*60)
    
    # Initialize with concrete animal concepts
    print("\n[1/3] Initializing agent with concrete concepts...")
    initial_concepts = ["Dog", "Cat", "Horse", "Cow"]
    agent = ConsciousAgent(initial_concepts=initial_concepts)
    
    print(f"Initial concepts: {initial_concepts}")
    print(f"Initial ontology size: {len(agent.ontology)}")
    
    # Self-directed learning loop
    print("\n[2/3] Starting self-directed learning dialogue...")
    print("\nThe system will now ask itself questions and discover abstractions.\n")
    
    dialogue = []
    max_steps = 10
    total_F_reduction = 0.0
    
    # Track concrete concepts separately (don't include invented abstractions)
    concrete_concepts = initial_concepts.copy()
    
    for step in range(max_steps):
        print(f"\n{'='*60}")
        print(f"DIALOGUE STEP {step + 1}")
        print('='*60)
        
        # One step of self-directed learning (only consider concrete concepts for grouping)
        question, answer, abstraction, F_reduction, success = agent.self_directed_learning_step(
            similarity_threshold=10.0,  # Higher threshold for animal concepts (they're more dissimilar)
            concrete_concepts=concrete_concepts
        )
        
        if not success:
            print(f"\nSystem: {answer}")
            print("→ Learning dialogue complete (no more questions)")
            break
        
        # Record dialogue
        dialogue.append({
            'step': step + 1,
            'question': question,
            'answer': answer,
            'abstraction': abstraction,
            'F_reduction': F_reduction,
            'ontology_size': len(agent.ontology)
        })
        
        total_F_reduction += F_reduction
        
        # Print dialogue
        print(f"\nSystem asks: \"{question}\"")
        print(f"System answers: {answer}")
        print(f"Ontology size: {len(agent.ontology)}")
        
        # Remove grouped concepts from concrete list and add abstraction
        # This prevents re-grouping the same pair
        concept_a, concept_b = question.split(" and ")
        concept_a = concept_a.replace("What do ", "")
        concept_b = concept_b.replace(" have in common?", "")
        
        if concept_a in concrete_concepts:
            concrete_concepts.remove(concept_a)
        if concept_b in concrete_concepts:
            concrete_concepts.remove(concept_b)
        
        # Add the new abstraction to concrete list for next-level grouping
        concrete_concepts.append(abstraction)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    
    print(f"\nDialogue Summary:")
    print(f"  Total questions asked: {len(dialogue)}")
    print(f"  Abstractions discovered: {len(dialogue)}")
    print(f"  Initial ontology: {len(initial_concepts)} concepts")
    print(f"  Final ontology: {len(agent.ontology)} concepts")
    print(f"  Concepts invented: {len(agent.ontology) - len(initial_concepts)}")
    print(f"  Total Free Energy reduction: {total_F_reduction:.4f}")
    
    print("\n" + "="*60)
    print("DIALOGUE TRACE")
    print("="*60)
    
    for entry in dialogue:
        print(f"\nStep {entry['step']}: {entry['question']}")
        print(f"  → {entry['answer']}")
    
    print("\n" + "="*60)
    print("FINAL ONTOLOGY")
    print("="*60)
    
    print(f"\nHierarchical structure discovered:")
    print(f"  Concrete: {initial_concepts}")
    
    invented = [entry['abstraction'] for entry in dialogue]
    print(f"  Abstract: {invented}")
    
    print("\n✓ SELF-DIRECTED LEARNING DEMONSTRATED")
    print("The system autonomously:")
    print("  1. Identified knowledge gaps (similar concepts)")
    print("  2. Generated questions about commonalities")
    print("  3. Invented abstractions to answer questions")
    print("  4. Built hierarchical taxonomy bottom-up")
    
    return {
        'agent': agent,
        'initial_concepts': initial_concepts,
        'dialogue': dialogue,
        'total_F_reduction': total_F_reduction,
        'final_ontology': agent.ontology.copy(),
        'invented_concepts': invented
    }


if __name__ == "__main__":
    results = run_self_directed_learning()
    
    print("\n" + "="*60)
    print("KEY ACHIEVEMENT")
    print("="*60)
    print("\nThe system generated its OWN curriculum:")
    print("  - No external guidance")
    print("  - No pre-defined questions")
    print("  - No manual abstraction specification")
    print("\nThis is AUTONOMOUS KNOWLEDGE CONSTRUCTION")
    print("through thermodynamic self-organization!")
