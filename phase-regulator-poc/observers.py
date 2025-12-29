import numpy as np
import time

EPS = 1e-9

class SimulationModel:
    """
    Mock LLM that provides deterministic metrics for PoC.
    """
    def __init__(self):
        self.knowledge_base = {
            "the capital of france is": "Paris",
            "the capital of germany is": "Berlin",
            "write a poem about a robot drinking tea": "In a kitchen of chrome and steel...",
        }

    def generate(self, prompt):
        prompt_lower = prompt.lower().strip()
        answer = "I'm not sure about that."
        
        # Determine metrics based on query type
        if "capital" in prompt_lower:
            # Factual queries are stable
            dO = 15.0 + np.random.normal(0, 0.2)
            # Find closest match in knowledge base
            for k in self.knowledge_base:
                if k in prompt_lower:
                    answer = self.knowledge_base[k]
                    break
            time.sleep(0.05) # Simulate inference time
        elif "poem" in prompt_lower:
            # Creative queries have high internal entropy (low dO)
            dO = 2.0 + np.random.normal(0, 0.5)
            answer = self.knowledge_base["write a poem about a robot drinking tea"]
            time.sleep(0.1) # Simulate longer inference
        else:
            # Unknown queries
            dO = 5.0 + np.random.normal(0, 1.0)
            time.sleep(0.02)
            
        return answer, dO

def estimate_delta_W(prompt, attempt_num=0):
    """
    Simulates semantic stability (witness consistency).
    """
    prompt_lower = prompt.lower().strip()
    if "capital" in prompt_lower:
        # High consistency for facts, close to dO
        return 16.0 + np.random.normal(0, 0.2)
    elif "poem" in prompt_lower:
        # Lower consistency for creative tasks
        return 4.0 + np.random.normal(0, 1.0)
    
    return 8.0

def compute_coupling(delta_O, delta_W):
    return (delta_O - delta_W) ** 2

def compute_hallucination_index(delta_O, delta_W):
    W_cross = compute_coupling(delta_O, delta_W)
    return W_cross / (delta_O * delta_W + EPS)

def extract_witness(text):
    """
    In simulation, we use a simple hash/string as the witness.
    """
    return text.lower().strip()
