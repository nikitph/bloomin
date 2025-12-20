import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import wikipediaapi

from semantic_wave_retrieval.engine import WaveRetrievalEngine
from semantic_wave_retrieval.rag_controller import WaveRAGController

# Configuration (Same as demo_concepts)
CONCEPTS = ["Photosynthesis", "French Revolution", "Quantum entanglement"]
QUERIES = {
    "Photosynthesis": [
        "How do plants convert sunlight into energy?", # Clean
        "pl4nt light r3action something", # Corrupted
        "Photosynthesis in animals and mammals energy process" # Adversarial
    ],
    "French Revolution": [
        "What caused the uprising in France in 1789?",
        "fr3nch rev0lut1on guillot1ne",
        "French Revolution in England and Germany 1999"
    ]
}

class WaveRAGDemo:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.wiki = wikipediaapi.Wikipedia('BloominBot/1.0', 'en')
        
    def setup(self):
        print("Fetching Wikipedia Data...")
        self.corpus = []
        for concept in CONCEPTS:
            page = self.wiki.page(concept)
            if page.exists():
                text = page.text
                # Simple chunking
                chunks = [p for p in text.split('\n') if len(p) > 50][:30]
                self.corpus.extend(chunks)
                print(f"Loaded {len(chunks)} chunks for {concept}")
        
        print("Embedding...")
        embeddings = self.model.encode(self.corpus)
        self.data_tensor = torch.tensor(embeddings, dtype=torch.float32)
        
        print("Initializing Wave Engine...")
        self.engine = WaveRetrievalEngine(self.data_tensor, k_neighbors=10, use_cuda=torch.cuda.is_available())
        
        print("Initializing RAG Controller...")
        self.controller = WaveRAGController(self.engine, self.corpus)
        
    def run(self):
        print("\n=== Wave-RAG End-to-End Demo ===")
        print(f"DeepSeek API Key Present: {'Yes' if os.environ.get('DEEPSEEK_API_KEY') else 'No'}")
        
        for concept, queries in QUERIES.items():
            print(f"\nTarget Concept: {concept}")
            for q in queries:
                print(f"\nQuery: {q}")
                q_vec = self.model.encode([q])[0]
                q_vec_t = torch.tensor(q_vec).to(self.device)
                
                result = self.controller.retrieve_and_generate(q, q_vec_t, tau=0.01) # Low tau for demo to ensure context
                
                print(f"  > Refused: {result['refusal']}")
                if not result['refusal']:
                    top_b = result['basins'][0] if result['basins'] else None
                    conf = top_b['confidence'] if top_b else 0.0
                    print(f"  > Grounded Basin Confidence: {conf:.4f}")
                    print(f"  > Context Snippet: {result['context'][:100]}...")
                    print(f"  > LLM Answer: {result['answer']}")
                else:
                    print(f"  > {result['answer']}")

if __name__ == "__main__":
    demo = WaveRAGDemo()
    demo.setup()
    demo.run()
