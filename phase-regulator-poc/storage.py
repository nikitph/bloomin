from dataclasses import dataclass
import time

@dataclass
class FrozenWitness:
    prompt: str
    key_witness: str        # simplified string witness for simulation
    value_witness: str        # the answer
    confidence_mass: float
    timestamp: float

class ThermalBloom:
    def __init__(self, match_threshold=0.8):
        self.witnesses = []
        self.MATCH_THRESHOLD = match_threshold

    def add(self, witness: FrozenWitness):
        # In simulation, we check for string equality or simple overlap
        for existing in self.witnesses:
            if existing.key_witness == witness.key_witness:
                existing.value_witness = witness.value_witness
                existing.confidence_mass = min(1.0, existing.confidence_mass + 0.1)
                existing.timestamp = witness.timestamp
                return
        self.witnesses.append(witness)

    def query(self, prompt_witness: str):
        # Simple exact match for simulation
        for witness in self.witnesses:
            if witness.key_witness == prompt_witness:
                return witness, "STRONG"
        
        # Simple prefix match for partial
        for witness in self.witnesses:
            if prompt_witness.startswith(witness.key_witness) or witness.key_witness.startswith(prompt_witness):
                return witness, "WEAK"
                
        return None, "NO_MATCH"

    def decay(self, current_time, tau_hours=72):
        tau_seconds = tau_hours * 3600
        new_witnesses = []
        for w in self.witnesses:
            elapsed = current_time - w.timestamp
            # Linear decay for simulation simplicity
            decay_factor = max(0.0, 1.0 - (elapsed / tau_seconds))
            w.confidence_mass *= decay_factor
            
            if w.confidence_mass > 0.1:
                new_witnesses.append(w)
        
        self.witnesses = new_witnesses
        return len(new_witnesses)
