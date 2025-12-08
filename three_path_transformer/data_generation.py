import torch
import numpy as np
from collections import defaultdict
from utils import random_peaked_distribution, geometric_mean

def generate_color_dataset(n_samples=1000, dim=128, max_depth=2):
    """
    Generate training examples for concept learning up to max_depth.
    Returns: 
        train_examples: List of {type, input1, input2, target}
        concepts: Dict of ground truth distributions
        hierarchy: Dict of depth -> list of names
    """
    
    # 1. GENERATE GROUND TRUTH CONCEPTS RECURSIVELY
    concepts = {}
    hierarchy = defaultdict(list)
    relationships = {} # child -> (p1, p2)
    
    # Base Layer: Primaries (Depth 0)
    # 3 Primaries is standard, but for depth 5 we might want more base diversity?
    # No, 3 primaries can generate 3C2 = 3 secondaries, 3+3=6 concepts.
    # Secondaries = 3. Tertiaries = pairs of secondaries...
    # Level 0: 3
    # Level 1: 3 choose 2 = 3 (Purple, Orange, Green)
    # Level 2: 3 choose 2 = 3 (Mauve, Chartreuse, Teal)
    # ... It stays at 3 if we only mix neighbors.
    # To get explosion, we must allow mixing ANY pair from level N? or cross-levels?
    # PRD implies strictly hierarchical: "Level 1: Red+Blue... Level 2: Purple+Orange".
    # If we stick to a ring of 3, we only get 3 unique children at each level (mixing neighbors).
    # That is sufficient to test depth.
    
    names_by_depth = {
        0: ['Red', 'Blue', 'Yellow'],
        1: ['Purple', 'Orange', 'Green'],
        2: ['Mauve', 'Chartreuse', 'Teal'],
        3: ['D3_A', 'D3_B', 'D3_C'], # Generic names for deeper levels
        4: ['D4_A', 'D4_B', 'D4_C'],
        5: ['D5_A', 'D5_B', 'D5_C']
    }
    
    # Initialize Primaries
    for i, name in enumerate(names_by_depth[0]):
        # Spaced peaks 
        loc = int(dim * ((i+0.5)/3.0))
        concepts[name] = random_peaked_distribution(dim=dim, peak_loc=loc)
        hierarchy[0].append(name)
        
    # Recursive Generation
    for d in range(1, max_depth + 1):
        parents = names_by_depth[d-1]
        children = names_by_depth[d]
        
        # Mix neighbors in the list (Ring topology)
        # (0,1), (1,2), (2,0)
        for i in range(len(parents)):
            p1_name = parents[i]
            p2_name = parents[(i+1) % len(parents)]
            child_name = children[i] # Assign Nth child to mix of N, N+1
            
            # Geometric Mean
            p1 = concepts[p1_name]
            p2 = concepts[p2_name]
            mix = geometric_mean(p1, p2)
            
            concepts[child_name] = mix
            hierarchy[d].append(child_name)
            relationships[child_name] = (p1_name, p2_name)
            
    all_concepts = concepts
    
    # 2. GENERATE TRAINING EXAMPLES
    examples = []
    
    mix_pairs = []
    for child, (p1, p2) in relationships.items():
        mix_pairs.append((p1, p2, child))
    
    # Generate Samples
    for _ in range(n_samples):
        # Sample a mixing pair
        c1, c2, target = mix_pairs[np.random.randint(len(mix_pairs))]
        
        examples.append({
            'type': 'mix',
            'input1': c1,
            'input2': c2,
            'target': all_concepts[target],
            'target_name': target
        })
        
    # Identity samples (Primaries + maybe deeper ones too)
    for _ in range(n_samples // 3):
        # Sample ANY concept for identity
        names = list(all_concepts.keys())
        c = names[np.random.randint(len(names))]
        examples.append({
            'type': 'identity',
            'input1': c,
            'input2': c,
            'target': all_concepts[c],
            'target_name': c
        })
        
    return examples, all_concepts, hierarchy, relationships

def verify_dataset(examples):
    print(f"Generated {len(examples)} examples")
    print("Sample example:", examples[0]['input1'], "+", examples[0]['input2'], "->", examples[0]['target_name'])
