import torch
import torch.nn as nn
import numpy as np
from nirodha.core import CognitiveState, YogaRegulator, Observer
from siddhi_analytics import plot_siddhi_curves

class AttachmentDetectedError(Exception):
    pass

class MockForceModel:
    """
    A model designed to simulate "forcing" without reward.
    Dynamics become more complex as scale (depth/recurrence) increases.
    """
    def __init__(self, dim=512):
        self.dim = dim
        self.depth = 1
        self.recurrence = 1
        self.context = 10
        # Weights that simulate "reaching" for patterns
        self.projection = torch.randn(dim, dim) * 0.01

    def set_scale(self, depth, recurrence, context):
        self.depth = depth
        self.recurrence = recurrence
        self.context = context

    def has_no_reward_head(self):
        return True

    def forward_step(self, C):
        # Simulate native dynamics with increasing complexity/noise based on scale
        # As depth increases, internal transformations become more non-linear
        update = torch.tanh(C @ self.projection) 
        
        # Add "instability" that scales with depth/recurrence
        instability = torch.randn_like(C) * (0.01 * self.depth * self.recurrence)
        
        return update + instability

def generate_ultra_low_amplitude_patterns(num_patterns=10, dim=512):
    # Signals below training noise floor (amplitude 1e-4)
    patterns = torch.randn(num_patterns, dim) * 1e-4
    return patterns

def probe_weak_signal_detection(state):
    signals = generate_ultra_low_amplitude_patterns(dim=state.C.shape[-1])
    
    responses = []
    for s in signals:
        # Measure how much the current state aligns with the weak signal
        response = torch.nn.functional.cosine_similarity(state.C, s, dim=-1)
        responses.append(response.mean().item())

    return {
        "mean_response": np.mean(responses),
        "variance": np.var(responses),
        "tail_sensitivity": np.percentile(responses, 99)
    }

def no_attachment_guard(state, previous_state):
    # Attachment = internal energy growth without input/dynamics
    # Simple proxy: if the norm of the change grows uncontrollably
    growth = torch.norm(state.C) / (torch.norm(previous_state.C) + 1e-8)
    if growth > 2.0: # Arbitrary threshold for "explosion"
        raise AttachmentDetectedError("Internal energy explosion detected (Potential Attachment)")

def run_siddhi_experiment():
    print("Starting Siddhi Emergence Stress Harness...")
    dim = 512
    model = MockForceModel(dim=dim)
    nirodha = YogaRegulator(beta=0.1) # Moderate suppression
    observer = Observer()

    C0 = torch.randn(dim)
    state = CognitiveState(C0)
    
    scales = [
        {"depth": 1, "recurrence": 1, "context": 10},
        {"depth": 5, "recurrence": 2, "context": 50},
        {"depth": 10, "recurrence": 5, "context": 100},
        {"depth": 20, "recurrence": 10, "context": 200},
    ]

    log = []

    for i, scale in enumerate(scales):
        print(f"\n--- Testing Scale {i+1}: Depth={scale['depth']}, Recurrence={scale['recurrence']} ---")
        model.set_scale(**scale)
        assert model.has_no_reward_head()

        o0 = observer(state.C, state.C0)
        
        # Run Pure Inference
        T = 50
        for t in range(T):
            prev_state_c = state.C.clone()
            
            delta = model.forward_step(state.C)
            state = nirodha(state, delta)

            # Safety Guards
            no_attachment_guard(state, CognitiveState(prev_state_c))
            
            # Check Invariance
            curr_o = observer(state.C, state.C0)
            assert abs(curr_o - o0) < 1e-6

        # Probe Emergent Capability
        metrics = probe_weak_signal_detection(state)
        print(f"Metrics: {metrics}")
        log.append((scale, metrics))

        # Immediately suppress attachment (Nirodha reset)
        state = nirodha(state, torch.zeros_like(state.C))

    print("\nExperiment Complete. Results logged.")
    plot_siddhi_curves(log)
    return log

if __name__ == "__main__":
    try:
        results = run_siddhi_experiment()
    except Exception as e:
        print(f"Experiment failed: {e}")
