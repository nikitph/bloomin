import torch
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator, Observer
from siddhi_harness import MockForceModel, probe_weak_signal_detection

def run_phase_map():
    print("Starting Phase Boundary Mapping...")
    betas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    scales = [
        {"depth": 1, "recurrence": 1, "context": 10},
        {"depth": 2, "recurrence": 2, "context": 20},
        {"depth": 4, "recurrence": 2, "context": 40},
        {"depth": 8, "recurrence": 4, "context": 80},
        {"depth": 16, "recurrence": 8, "context": 160},
        {"depth": 32, "recurrence": 16, "context": 320},
    ]
    
    dim = 512
    onset_threshold = 0.005 # Sensitivity threshold for "onset"
    
    results = {} # beta -> {onset_scale: x, tail_sensitivity: y}

    for beta in betas:
        print(f"\nEvaluating beta={beta}")
        model = MockForceModel(dim=dim)
        nirodha = YogaRegulator(beta=beta)
        
        C0 = torch.randn(dim)
        state = CognitiveState(C0)
        
        onset_scale = None
        final_tail_sensitivity = 0
        
        for i, scale in enumerate(scales):
            model.set_scale(**scale)
            
            # Run Inference
            T = 30
            for t in range(T):
                delta = model.forward_step(state.C)
                state = nirodha(state, delta)
            
            # Probe
            metrics = probe_weak_signal_detection(state)
            
            effective_scale = scale['depth'] * scale['recurrence']
            if onset_scale is None and metrics['mean_response'] > onset_threshold:
                onset_scale = effective_scale
                print(f"  Siddhi Onset detected at scale {onset_scale}")
            
            final_tail_sensitivity = metrics['tail_sensitivity']
            
            # Suppression reset
            state = nirodha(state, torch.zeros_like(state.C))

        results[beta] = {
            "onset_scale": onset_scale if onset_scale is not None else max([s['depth']*s['recurrence'] for s in scales]) * 2,
            "tail_sensitivity": final_tail_sensitivity
        }

    # Plotting
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(betas, [results[b]['onset_scale'] for b in betas], marker='o', color='blue')
    plt.xscale('log')
    plt.title('Phase Boundary: Onset Scale vs. Beta')
    plt.xlabel('Beta (Suppression Strength)')
    plt.ylabel('Onset Scale (Depth * Recurrence)')
    plt.grid(True, which="both", ls="-")

    plt.subplot(2, 1, 2)
    plt.plot(betas, [results[b]['tail_sensitivity'] for b in betas], marker='s', color='red')
    plt.xscale('log')
    plt.title('Final Tail Sensitivity vs. Beta')
    plt.xlabel('Beta (Suppression Strength)')
    plt.ylabel('99th Percentile Sensitivity')
    plt.grid(True, which="both", ls="-")

    plt.tight_layout()
    plt.savefig('siddhi_phase_diagram.png')
    print("\nPhase diagram saved to siddhi_phase_diagram.png")
    return results

if __name__ == "__main__":
    run_phase_map()
