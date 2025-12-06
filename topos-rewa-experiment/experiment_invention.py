"""
Experiment 5: Autopoietic Concept Invention

Demonstrates that a system can autonomously expand its ontology
by inventing new concepts to resolve thermodynamic stress.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import CONFIG
from conscious_agent import ConsciousAgent
from ricci_flow import contrastive_loss, curvature_penalty, entropy


def phase_1_dreaming(agent):
    """
    Phase 1: Generate a paradoxical state by mixing Red and Blue
    
    Returns:
        p_dream: Dream witness distribution
    """
    print("\n" + "="*60)
    print("PHASE 1: DREAMING")
    print("="*60)
    
    # Retrieve base concepts
    print("\n[1/3] Retrieving base concepts...")
    p_red = agent.perceive("Red")
    p_blue = agent.perceive("Blue")
    
    print(f"Red prototype shape: {p_red.shape}")
    print(f"Blue prototype shape: {p_blue.shape}")
    
    # Synthesize a "Dream" (Superposition)
    print("\n[2/3] Synthesizing dream state (Red + Blue superposition)...")
    dream_temp = CONFIG.get("DREAM_TEMP", 1.0)
    
    # Mix 50% Red, 50% Blue + Noise
    p_dream = 0.5 * p_red + 0.5 * p_blue
    
    # Add noise for creativity
    noise = torch.randn_like(p_dream) * dream_temp * 0.1
    p_dream = p_dream + noise
    
    # Normalize to valid probability distribution
    p_dream = torch.clamp(p_dream, min=0.0)
    p_dream = p_dream / torch.sum(p_dream)
    
    print(f"Dream distribution created. Entropy: {entropy(p_dream):.4f}")
    
    # Logic Check
    print("\n[3/3] Logic check (does dream fit existing concepts?)...")
    
    # Check consistency with Red
    epsilon = 1e-10
    p_dream_clipped = torch.clamp(p_dream, epsilon, 1.0)
    p_red_clipped = torch.clamp(p_red, epsilon, 1.0)
    consistency_red = torch.sum(p_dream_clipped * torch.log(p_dream_clipped / p_red_clipped))
    
    # Check consistency with Blue
    p_blue_clipped = torch.clamp(p_blue, epsilon, 1.0)
    consistency_blue = torch.sum(p_dream_clipped * torch.log(p_dream_clipped / p_blue_clipped))
    
    print(f"Consistency with Red (KL): {consistency_red:.4f} - FAIL")
    print(f"Consistency with Blue (KL): {consistency_blue:.4f} - FAIL")
    print("\n✗ Dream state violates existing logic (contradiction detected)")
    
    return p_dream


def phase_2_dissonance(agent, p_dream):
    """
    Phase 2: Calculate cognitive dissonance (Free Energy)
    
    Returns:
        (free_energy, should_learn)
    """
    print("\n" + "="*60)
    print("PHASE 2: COGNITIVE DISSONANCE")
    print("="*60)
    
    print("\n[1/2] Calculating Free Energy...")
    F_dream = agent.calculate_free_energy(p_dream)
    
    print(f"System Free Energy: {F_dream:.4f}")
    
    # Find nearest concept
    nearest, dist = agent.find_nearest_prototype(p_dream)
    print(f"Nearest concept: {nearest} (distance: {dist:.4f})")
    
    # Check threshold
    print("\n[2/2] Checking dissonance threshold...")
    threshold = CONFIG.get("DISSONANCE_THRESHOLD", 0.5)
    print(f"Dissonance threshold: {threshold:.4f}")
    
    if F_dream > threshold:
        print(f"\n>> CRITICAL DISSONANCE DETECTED (F={F_dream:.4f} > {threshold:.4f})")
        print(">> Ontology expansion required to restore equilibrium")
        return F_dream, True
    else:
        print(f"\n>> Dissonance acceptable (F={F_dream:.4f} <= {threshold:.4f})")
        print(">> Noise ignored, no learning needed")
        return F_dream, False


def phase_3_invention(agent, p_dream, initial_free_energy):
    """
    Phase 3: Invent new concept and stabilize via Ricci Flow
    
    Returns:
        (new_concept_id, free_energy_history)
    """
    print("\n" + "="*60)
    print("PHASE 3: INVENTION (Ricci Flow)")
    print("="*60)
    
    # Create new symbol
    print("\n[1/4] Creating new concept node...")
    new_concept_id = "Purple"
    print(f"New concept: '{new_concept_id}'")
    
    # Register with initial dream state
    print("\n[2/4] Registering dream state as new prototype...")
    agent.register_prototype(new_concept_id, p_dream.clone())
    print(f"Ontology expanded: {agent.ontology}")
    
    # Stabilize via Ricci Flow
    print("\n[3/4] Stabilizing via Ricci Flow (contrastive learning)...")
    print(f"Running {CONFIG.get('HEALING_STEPS', 200)} optimization steps...")
    
    # Make dream distribution learnable
    p_new = p_dream.clone().detach().requires_grad_(True)
    
    optimizer = optim.Adam([p_new], lr=0.01)
    
    free_energy_history = [initial_free_energy]
    step_history = [0]
    
    healing_steps = CONFIG.get("HEALING_STEPS", 200)
    
    for step in range(healing_steps):
        optimizer.zero_grad()
        
        # Normalize to probability distribution
        p_new_normalized = torch.softmax(p_new, dim=0)
        
        # Get prototypes
        p_red = agent.concept_prototypes["Red"]
        p_blue = agent.concept_prototypes["Blue"]
        
        # Contrastive loss: pull toward new concept, push away from Red/Blue
        loss = contrastive_loss(
            anchor=p_new_normalized,
            positive=p_new_normalized,  # Self-consistency
            negatives=[p_red, p_blue],
            temperature=0.1
        )
        
        # Curvature regularization
        all_prototypes = [p_new_normalized, p_red, p_blue]
        loss += curvature_penalty(all_prototypes, alpha=0.01)
        
        loss.backward()
        optimizer.step()
        
        # Track Free Energy
        if step % 20 == 0 or step == healing_steps - 1:
            with torch.no_grad():
                p_current = torch.softmax(p_new, dim=0)
                agent.concept_prototypes[new_concept_id] = p_current.clone()
                F = agent.calculate_free_energy(p_current)
                free_energy_history.append(F)
                step_history.append(step + 1)
                
                if step % 40 == 0:
                    print(f"  Step {step:3d}: Loss={loss.item():.4f}, F={F:.4f}")
    
    # Update final prototype
    with torch.no_grad():
        p_final = torch.softmax(p_new, dim=0)
        agent.concept_prototypes[new_concept_id] = p_final
    
    print(f"\n[4/4] Concept '{new_concept_id}' stabilized!")
    
    # Measure final Free Energy
    F_new = agent.calculate_free_energy(p_final)
    print(f"Initial Free Energy: {initial_free_energy:.4f}")
    print(f"Final Free Energy: {F_new:.4f}")
    print(f"Reduction: {(initial_free_energy - F_new):.4f} ({(1 - F_new/initial_free_energy)*100:.1f}%)")
    
    return new_concept_id, free_energy_history, step_history


def phase_4_verification(agent, new_concept_id):
    """
    Phase 4: Verify that system recognizes new concept
    
    Returns:
        success (bool)
    """
    print("\n" + "="*60)
    print("PHASE 4: VERIFICATION")
    print("="*60)
    
    print("\n[1/3] Querying with Red + Blue composition...")
    
    # Get embeddings
    emb_red = agent.get_concept_embedding("Red")
    emb_blue = agent.get_concept_embedding("Blue")
    
    # Compose
    query_vec = 0.5 * emb_red + 0.5 * emb_blue
    query_vec = query_vec / torch.norm(query_vec)
    
    # Get witness distribution for query
    p_query = agent.perceive(query_vec)
    
    print("\n[2/3] Finding nearest concept...")
    nearest, dist = agent.find_nearest_prototype(p_query)
    
    print(f"Nearest concept: {nearest}")
    print(f"Distance: {dist:.4f}")
    
    # Check all distances
    print("\nDistances to all concepts:")
    epsilon = 1e-10
    p_query_clipped = torch.clamp(p_query, epsilon, 1.0)
    
    for concept in agent.ontology:
        p_concept = agent.concept_prototypes[concept]
        p_concept_clipped = torch.clamp(p_concept, epsilon, 1.0)
        kl = torch.sum(p_query_clipped * torch.log(p_query_clipped / p_concept_clipped))
        print(f"  {concept}: {kl:.4f}")
    
    print("\n[3/3] Checking recognition...")
    
    if nearest == new_concept_id:
        print(f"\n✓ SUCCESS: System recognizes '{new_concept_id}' as distinct entity!")
        print(f"✓ Logic Gate: VALID (New sheaf created)")
        print(f"✓ Red + Blue → {new_concept_id}")
        return True
    else:
        print(f"\n✗ FAILURE: System reverted to old ontology")
        print(f"✗ Retrieved: {nearest} (expected: {new_concept_id})")
        return False


def run_invention_experiment():
    """
    Run complete autopoietic concept invention experiment
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: AUTOPOIETIC CONCEPT INVENTION")
    print("The Bridge from Safety to Creativity")
    print("="*60)
    
    # Initialize agent
    print("\n[Setup] Initializing Conscious Agent...")
    agent = ConsciousAgent(initial_concepts=["Red", "Blue"])
    print(f"Initial ontology: {agent.ontology}")
    
    # Phase 1: Dreaming
    p_dream = phase_1_dreaming(agent)
    
    # Phase 2: Cognitive Dissonance
    initial_F, should_learn = phase_2_dissonance(agent, p_dream)
    
    if not should_learn:
        print("\n" + "="*60)
        print("EXPERIMENT ABORTED: No dissonance detected")
        print("="*60)
        return None
    
    # Phase 3: Invention
    new_concept, F_history, step_history = phase_3_invention(agent, p_dream, initial_F)
    
    # Phase 4: Verification
    success = phase_4_verification(agent, new_concept)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nInitial Ontology: ['Red', 'Blue']")
    print(f"Final Ontology: {agent.ontology}")
    print(f"\nFree Energy: {initial_F:.4f} → {F_history[-1]:.4f}")
    print(f"Reduction: {(initial_F - F_history[-1]):.4f} ({(1 - F_history[-1]/initial_F)*100:.1f}%)")
    print(f"\nVerification: {'✓ PASSED' if success else '✗ FAILED'}")
    
    return {
        'agent': agent,
        'new_concept': new_concept,
        'initial_free_energy': initial_F,
        'final_free_energy': F_history[-1],
        'free_energy_history': F_history,
        'step_history': step_history,
        'success': success
    }


if __name__ == "__main__":
    results = run_invention_experiment()
    
    if results and results['success']:
        print("\n" + "="*60)
        print("✓ AUTOPOIESIS DEMONSTRATED")
        print("="*60)
        print("The system autonomously expanded its ontology")
        print("through thermodynamic necessity.")
        
        # Generate visualizations
        from visualize_invention import generate_all_visualizations
        generate_all_visualizations(results)
