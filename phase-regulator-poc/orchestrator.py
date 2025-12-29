import time
from observers import SimulationModel, estimate_delta_W, compute_hallucination_index, extract_witness
from regulator import SpectralState, PhaseRegulator
from storage import ThermalBloom, FrozenWitness

class PhaseAwareInference:
    def __init__(self):
        print("Initializing Simulation Engine...")
        self.model = SimulationModel()
        self.bloom = ThermalBloom()
        self.regulator = PhaseRegulator(H_threshold=0.2, freeze_T=3, melt_T=1)
        self.cluster_states = {}

    def get_cluster(self, prompt):
        return " ".join(prompt.lower().split()[:3])

    def answer_query(self, prompt):
        cluster = self.get_cluster(prompt)
        if cluster not in self.cluster_states:
            self.cluster_states[cluster] = SpectralState()
        
        state = self.cluster_states[cluster]
        prompt_witness = extract_witness(prompt)
        
        # 1. Query L0
        frozen, match_type = self.bloom.query(prompt_witness)
        
        if frozen and state.phase == "SOLID" and match_type == "STRONG":
            return frozen.value_witness, "L0 (SOLID)", 0.0001, 0.0, 0.0

        # 2. Simulation Reasoning (L3)
        start_time = time.time()
        answer, dO = self.model.generate(prompt)
        latency = time.time() - start_time
        
        # 3. Metrics
        dW = estimate_delta_W(prompt)
        H = compute_hallucination_index(dO, dW)
        
        # 4. Update Regulator
        self.regulator.update_metrics(state, dO, dW, H)
        change = self.regulator.transition(state)
        
        if "FREEZE" in change:
            witness = FrozenWitness(
                prompt=prompt,
                key_witness=prompt_witness,
                value_witness=answer,
                confidence_mass=1.0,
                timestamp=time.time()
            )
            self.bloom.add(witness)
            print(f"---> PHASE TRANSITION: {change} for cluster '{cluster}'")
        elif "MELT" in change or "HYBRID" in change:
            print(f"---> PHASE TRANSITION: {change} for cluster '{cluster}'")

        return answer, f"L3 ({state.phase})", latency, dO, dW
