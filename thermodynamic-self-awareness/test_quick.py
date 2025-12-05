# -*- coding: utf-8 -*-
"""
Quick Test Script

Runs a minimal version of the self-discovery experiment to verify
all modules are working correctly.
"""

from __future__ import print_function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from modules import REWAMemory, ToposLayer, RicciFlow, SemanticRG, AGIController
from datasets import create_hidden_rule_dataset


def test_modules():
    """Test that all modules can be instantiated"""
    print("Testing module instantiation...")
    
    memory = REWAMemory(K=8, L=16, seed=42)
    topos = ToposLayer(kl_threshold=0.1)
    ricci = RicciFlow()
    rg = SemanticRG()
    controller = AGIController(seed=42)
    
    print("[OK] All modules instantiated successfully")
    return memory, topos, ricci, rg, controller


def test_dataset():
    """Test dataset generation"""
    print("\nTesting dataset generation...")
    
    train, test, queries = create_hidden_rule_dataset(
        n_train=100,
        n_test=20,
        seed=42
    )
    
    print("[OK] Generated {} training examples".format(len(train)))
    print("[OK] Generated {} test examples".format(len(test)))
    print("[OK] Generated {} test queries".format(len(queries)))
    
    # Check rule is present
    red_count = sum(1 for ex in train if ex.attributes['color'] == 'red')
    red_large_count = sum(
        1 for ex in train 
        if ex.attributes['color'] == 'red' and ex.hidden_properties['size'] == 'large'
    )
    
    print("  Red items: {}".format(red_count))
    print("  Red + Large: {}".format(red_large_count))
    print("  Rule strength: {:.1%}".format(float(red_large_count) / red_count))
    
    return train, test, queries


def test_conscious_cycle(memory, topos, ricci, rg, controller, train_examples):
    """Test a single conscious cycle"""
    print("\nTesting conscious cycle...")
    
    # Load a few examples
    for example in train_examples[:50]:
        ws = memory.extract_witnesses(example.features, item_id=example.item_id)
        ws.metadata = {'attributes': example.attributes}
        memory.store(ws)
    
    print("[OK] Loaded {} items".format(memory.get_statistics()['num_items']))
    
    # Run one cycle
    diagnostics = controller.conscious_cycle(
        rewa_memory=memory,
        topos_layer=topos,
        ricci_flow=ricci,
        semantic_rg=rg,
        external_input=None
    )
    
    print("[OK] Conscious cycle completed")
    print("  Free Energy: {:.4f}".format(diagnostics.free_energy))
    print("  Semantic Energy: {:.4f}".format(diagnostics.semantic_energy))
    print("  Curvature Entropy: {:.4f}".format(diagnostics.curvature_entropy))
    print("  Contradictions: {}".format(diagnostics.contradictions_detected))
    print("  Ricci Updates: {}".format(diagnostics.ricci_updates))
    
    return diagnostics


def test_mini_experiment():
    """Run a mini experiment (10 epochs)"""
    print("\n" + "="*60)
    print("Running Mini Experiment (10 epochs)")
    print("="*60)
    
    # Setup
    memory = REWAMemory(K=16, L=32, seed=42)
    topos = ToposLayer(kl_threshold=0.1)
    ricci = RicciFlow()
    rg = SemanticRG()
    controller = AGIController(temperature=0.1, query_budget=10, seed=42)
    
    # Load data
    train, _, _ = create_hidden_rule_dataset(n_train=500, seed=42)
    
    for example in train:
        ws = memory.extract_witnesses(example.features, item_id=example.item_id)
        ws.metadata = {'attributes': example.attributes}
        memory.store(ws)
    
    print("Loaded {} items\n".format(memory.get_statistics()['num_items']))
    
    # Run cycles
    free_energies = []
    
    for epoch in range(10):
        diagnostics = controller.conscious_cycle(
            rewa_memory=memory,
            topos_layer=topos,
            ricci_flow=ricci,
            semantic_rg=rg,
            external_input=None
        )
        
        free_energies.append(diagnostics.free_energy)
        
        print("Epoch {}: F={:.4f}, Contradictions={}, Ricci={}".format(
            epoch, diagnostics.free_energy,
            diagnostics.contradictions_detected,
            diagnostics.ricci_updates
        ))
    
    # Check trend
    if len(free_energies) > 1:
        slope = np.polyfit(range(len(free_energies)), free_energies, 1)[0]
        print("\nFree energy slope: {:.6f}".format(slope))
        if slope < 0:
            print("[OK] Free energy is decreasing!")
        else:
            print("[WARN] Free energy is not decreasing (may need more epochs)")
    
    print("\n" + "="*60)
    print("Mini experiment complete!")
    print("="*60)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Thermodynamic Self-Awareness - Quick Test")
    print("="*60 + "\n")
    
    # Test 1: Module instantiation
    memory, topos, ricci, rg, controller = test_modules()
    
    # Test 2: Dataset generation
    train, test, queries = test_dataset()
    
    # Test 3: Single conscious cycle
    diagnostics = test_conscious_cycle(memory, topos, ricci, rg, controller, train)
    
    # Test 4: Mini experiment
    test_mini_experiment()
    
    print("\n[OK] All tests passed!")
    print("\nYou can now run the full experiment with:")
    print("  python experiments/experiment_a_self_discovery.py")
