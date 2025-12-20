import os
import requests
import json
from typing import List, Dict, Any

class WaveRAGController:
    def __init__(self, engine, corpus: List[str]):
        """
        Args:
            engine: WaveRetrievalEngine instance
            corpus: List of text chunks corresponding to engine.data indices
        """
        self.engine = engine
        self.corpus = corpus
        self.api_key = os.environ.get("DEEPSEEK_API_KEY") 
        # DeepSeek API endpoint (compatible with OpenAI Client if using openai lib, 
        # but let's use direct requests for transparency or openai lib if preferred)
        self.api_url = "https://api.deepseek.com/v1/chat/completions" # Hypothetical standard endpoint
        
    def retrieve_and_generate(self, query: str, 
                              query_vec, # Pre-encoded vector
                              tau: float = 0.05, # Confidence threshold
                              tau_high: float = 0.5): # High confidence threshold (not used in simple logic yet)
        
        # 1. Wave Retrieval
        basins = self.engine.retrieve_basins(query_vec, top_k_basins=3,
                                           wave_params={'T_wave': 15, 'c': 1.0, 'sigma': 1.5}, 
                                           telegrapher_params={'T_damp': 15, 'gamma': 0.8},
                                           poisson_params={'alpha': 0.1})
        
        # 2. Admission Rule / Refusal
        # Filter basins by confidence
        valid_basins = [b for b in basins if b['confidence'] >= tau]
        
        if not valid_basins:
            return {
                "answer": "I cannot answer this question with sufficient confidence based on the available knowledge.",
                "refusal": True,
                "basins": basins,
                "context": ""
            }
            
        # 3. Context Assembly
        # For MVP, take top 1 basin if valid, or merge top 2 if ambiguous?
        # Let's take the top valid basin.
        top_basin = valid_basins[0]
        
        # Get representative texts
        # In engine.py retrieve_basins, we currently return 'representatives': [id]
        # We should ideally get neighbors. For now, just the center node text.
        # TODO: Expand to k-nearest in that basin.
        context_ids = top_basin['representatives']
        context_text = "\n\n".join([f"[{doc_id}]: {self.corpus[doc_id]}" for doc_id in context_ids])
        
        # 4. LLM Generation
        prompt = self._construct_prompt(query, context_text, top_basin['confidence'])
        
        if self.api_key:
            answer = self._call_llm(prompt)
        else:
            answer = "[Mock LLM Output] Based on the context..."
            print("Warning: DEEPSEEK_API_KEY not found. Using Mock Output.")
            
        return {
            "answer": answer,
            "refusal": False,
            "basins": valid_basins,
            "context": context_text
        }
    
    def _construct_prompt(self, query: str, context: str, confidence: float) -> str:
        return f"""You are answering based only on the following grounded context.
The system identified the most relevant semantic basin with confidence {confidence:.2f}.

Context:
<<<
{context}
>>>

If the context is insufficient or contradictory, say so explicitly.

Question: {query}
Answer:"""

    def _call_llm(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat", # or deepseek-coder
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=data, timeout=10)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            else:
                return f"LLM Error: {resp.status_code} - {resp.text}"
        except Exception as e:
            return f"LLM Connection Failed: {e}"
