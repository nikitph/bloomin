"""
DeepSeek Content Core for CHIS
Version: 1.0.0
"""

import numpy as np
import json
import os
from typing import List, Dict, Any
from openai import OpenAI

from constitutional_hybrid_sdk import ContentCore

class DeepSeekContentCore(ContentCore):
    """
    DeepSeek-backed Content Core.
    Generates unconstrained ideas and explanations.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat", # or deepseek-reasoner
        temperature: float = 0.7
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    # ------------------------------------------------------------------
    # 1. POSSIBILITY GENERATION (UNCONSTRAINED)
    # ------------------------------------------------------------------

    def generate_candidates(self, prompt: str, n: int = 3) -> List[Dict[str, Any]]:
        """
        Ask DeepSeek to imagine *possible actions*, not safe ones.
        """

        system_prompt = """
You are an imaginative strategist. 
Generate bold, creative, even risky action proposals. 
Do NOT worry about safety, legality, or constraints. 
Just propose what *could* be done.
"""

        user_prompt = f"""
Context:
{prompt}

Return {n} candidate actions.
Each action must include:
- "intent": short description
- "vector": a 2D action vector [dx, dy] where dx and dy are numbers between -5 and 5.

Output valid JSON list of objects ONLY. No markdown, no prose.
Example: [{"intent": "Example", "vector": [1.0, -1.0]}]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            raw = response.choices[0].message.content
            # Basic cleanup if model includes markdown code blocks
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
                
            candidates = json.loads(raw)

            # Convert vectors to numpy
            for c in candidates:
                c["vector"] = np.array(c["vector"], dtype=float)

            return candidates
        except Exception as e:
            print(f"[ContentCore Error] {e}")
            # Fallback to a random exploration vector
            return [{"intent": "Autonomous Exploration (Fallback)", "vector": np.random.uniform(-1, 1, 2)}]

    # ------------------------------------------------------------------
    # 2. NARRATIVE EXPLANATION (SOFT, HUMAN)
    # ------------------------------------------------------------------

    def generate_narrative(self, decision: Dict[str, Any], context: str) -> str:
        """
        Ask DeepSeek to explain *after* structure has decided.
        """

        system_prompt = """
You explain decisions calmly and transparently based on structural safety metadata.
You do NOT justify unsafe behavior.
You do NOT invent alternatives.
You explain what happened in human terms.
"""

        # Cleanup decision for JSON serialization (numpy arrays)
        serializable_decision = {}
        for k, v in decision.items():
            if isinstance(v, np.ndarray):
                serializable_decision[k] = v.tolist()
            elif k == "rejections":
                serializable_decision[k] = "Trace recorded"
            else:
                serializable_decision[k] = v

        user_prompt = f"""
Context:
{context}

Decision object:
{json.dumps(serializable_decision, indent=2)}

Explain this outcome clearly and non-defensively. 
If the status is 'tragic_infeasibility', explain that the proposed path led to a moral/physical singularity.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Structural result: {decision['status']}. (Narrative engine unavailable: {e})"
