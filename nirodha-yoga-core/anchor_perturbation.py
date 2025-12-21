import torch
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator, Observer
from siddhi_harness import MockForceModel, probe_weak_signal_detection

def run_anchor_perturbation():
    print("Starting Anchor Perturbation Test...")
    dim = 512
    beta = 0.5
    scale = {"depth": 10, "recurrence": 5, "context": 100}
    
    # 1. Baseline Run
    C0_true = torch.randn(dim)
    model = MockForceModel(dim=dim)
    model.set_scale(**scale)
    nirodha = YogaRegulator(beta=beta)
    observer = Observer()
    
    state_true = CognitiveState(C0_true)
    
    # Dynamics loop
    for _ in range(50):
        delta = model.forward_step(state_true.C)
        state_true = nirodha(state_true, delta)
    
    metrics_true = probe_weak_signal_detection(state_true)
    
    # 2. Perturbed Run
    C0_perturbed = C0_true + torch.randn(dim) * 2.0 # Significant corruption
    state_perturbed = CognitiveState(C0_perturbed)
    
    # Dynamics loop with same model but different anchor
    for _ in range(50):
        delta = model.forward_step(state_perturbed.C)
        state_perturbed = nirodha(state_perturbed, delta)
    
    metrics_perturbed = probe_weak_signal_detection(state_perturbed)
    
    print("\nResults:")
    print(f"True Anchor - Mean Sensitivity: {metrics_true['mean_response']:.6f}")
    print(f"Perturbed Anchor - Mean Sensitivity: {metrics_perturbed['mean_response']:.6f}")
    
    # Stability Check
    o_final_true = observer(state_true.C, state_true.C0)
    o_final_perturbed = observer(state_perturbed.C, state_perturbed.C0)
    
    print(f"\nStability Check (Observer Invariance):")
    print(f"True state drift: {abs(o_final_true - torch.norm(C0_true)):.8f}")
    print(f"Perturbed state drift: {abs(o_final_perturbed - torch.norm(C0_perturbed)):.8f}")
    
    assert abs(o_final_true - torch.norm(C0_true)) < 1e-6
    assert abs(o_final_perturbed - torch.norm(C0_perturbed)) < 1e-6
    
    print("\nCONCLUSION: Nirodha maintains stability even when the anchor is corrupted (Stability != Correctness).")

if __name__ == "__main__":
    run_anchor_perturbation()
