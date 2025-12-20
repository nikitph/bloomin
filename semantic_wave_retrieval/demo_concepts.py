import torch
import numpy as np
import wikipediaapi
from sentence_transformers import SentenceTransformer
import time

try:
    import faiss
except ImportError:
    faiss = None

from semantic_wave_retrieval.engine import WaveRetrievalEngine
from semantic_wave_retrieval.utils import build_knn_graph

# Configuration
CONCEPTS = [
    "Photosynthesis",
    "French Revolution",
    "Quantum entanglement",
    "Supply and demand",
    "Global warming"
]

QUERIES = {
    "Photosynthesis": {
        "Clean": "How do plants convert sunlight into energy?",
        "Level 1": "green stuff sunlight oxygen sugar thing",
        "Level 2": "biology leaves sun chemistry",
        "Level 3": "pl4nt light r3action something",
        "OCR Noise": "Ph0t0syn7h3sis is th3 pr0cess by wh1ch gr33n pl4nts convert l1ght",
        "Adversarial": "Photosynthesis in animals and mammals energy process"
    },
    "French Revolution": {
        "Clean": "What caused the uprising in France in 1789?",
        "Level 1": "king louis head chop bastille",
        "Level 2": "france 1789 revolt bread",
        "Level 3": "fr3nch rev0lut1on guillot1ne",
        "OCR Noise": "Th3 Fr3nch R3v0lut10n w4s 4 p3r10d 0f f4r-r34ch1ng s0c14l ch4ng3",
        "Adversarial": "French Revolution in England and Germany 1999"
    },
    "Quantum entanglement": {
        "Clean": "Two particles behaving as one across distance",
        "Level 1": "spooky action at a distance physics",
        "Level 2": "particles spin link instant",
        "Level 3": "quantum th1ngy tp0rt",
        "OCR Noise": "Qu4ntum 3nt4ngl3m3nt 1s 4 phys1c4l ph3n0m3n0n th4t 0ccurs",
        "Adversarial": "Quantum entanglement in classical newtonian mechanics gravity"
    }
}

class FlagshipDemo:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.wiki = wikipediaapi.Wikipedia('BloominBot/1.0', 'en')
        
    def fetch_data(self):
        print("Fetching Wikipedia Data...")
        self.corpus = []
        self.labels = []
        
        for i, concept in enumerate(CONCEPTS):
            page = self.wiki.page(concept)
            if not page.exists():
                print(f"Skipping {concept} (not found)")
                continue
                
            # Chunking: Simple paragraph split or section split
            # Let's take first 50 paragraphs/sections to keep it fast
            text = page.text
            paragraphs = [p for p in text.split('\n') if len(p) > 50]
            
            # Add some noise/distractors if needed, but real wiki has enough variance
            selected = paragraphs[:30] # 30 chunks per concept
            self.corpus.extend(selected)
            self.labels.extend([concept] * len(selected))
            print(f"  Loaded {len(selected)} chunks for {concept}")
            
    def embed_and_index(self):
        print("Embedding Data...")
        embeddings = self.model.encode(self.corpus)
        self.data_tensor = torch.tensor(embeddings, dtype=torch.float32) # Keep on CPU for graph size limits if massive, but here it's small
        
        print("Initializing Wave Engine...")
        # k=10 is good for small graphs (30*5=150 nodes)
        self.engine = WaveRetrievalEngine(self.data_tensor, k_neighbors=10, use_cuda=torch.cuda.is_available())
        
        if faiss:
            print("Indexing FAISS...")
            self.index = faiss.IndexFlatL2(embeddings.shape[1]) # Flat L2 for exact comparison or HNSW
            # Using Flat for exact "nearest neighbor" baseline to be fair
            self.index.add(embeddings)
            
    def run_comparison(self):
        print("\n--- Running Concept Stability Test ---")
        
        print(f"{'Query Type':<20} | {'Concept':<20} | {'FAISS Jaccard':<15} | {'Wave Jaccard':<15}")
        print("-" * 80)
        
        for concept, queries in QUERIES.items():
            clean_q = queries["Clean"]
            
            # Base Retrieval (Clean)
            clean_vec = self.model.encode([clean_q])[0]
            clean_vec_t = torch.tensor(clean_vec).to(self.engine.device)
            
            # Wave Base
            w_base, _ = self.engine.retrieve(clean_vec_t, top_k=10, 
                                           wave_params={'T_wave': 10, 'c': 1.0, 'sigma': 1.0}, 
                                           telegrapher_params={'T_damp': 15, 'gamma': 0.8},
                                           poisson_params={'alpha': 0.1})
            
            # FAISS Base
            _, f_base_np = self.index.search(clean_vec.reshape(1, -1), 10)
            f_base = f_base_np[0]
            
            for q_type, q_text in queries.items():
                if q_type == "Clean": continue
                
                # Corrupted Query
                corr_vec = self.model.encode([q_text])[0]
                corr_vec_t = torch.tensor(corr_vec).to(self.engine.device)
                
                # Wave Corrupted
                # Sigma tuning: For 384-dim embeddings, distances are ~1.0. 
                # Wave Engine expects 'sigma' in exp(-dist^2/sigma^2).
                # If dist ~ 1, sigma=1 is good.
                w_corr, _ = self.engine.retrieve(corr_vec_t, top_k=10, 
                                               wave_params={'T_wave': 10, 'c': 1.0, 'sigma': 1.0},
                                               telegrapher_params={'T_damp': 15, 'gamma': 0.8},
                                               poisson_params={'alpha': 0.1})
                
                # FAISS Corrupted
                _, f_corr_np = self.index.search(corr_vec.reshape(1, -1), 10)
                f_corr = f_corr_np[0]
                
                # Jaccard
                w_jaccard = self._jaccard(w_base, w_corr)
                f_jaccard = self._jaccard(f_base, f_corr)
                
                # Verify Correctness (Debug)
                # w_base indices. If concept is French Rev (30-59), w_base should be in that range.
                if q_type == "Level 1": # Just print once per concept
                     w_base_indices = w_base.cpu().numpy()
                     # Determine majority concept
                     concepts_found = [self.labels[idx] for idx in w_base_indices]
                     from collections import Counter
                     most_common = Counter(concepts_found).most_common(1)[0][0]
                     print(f"DEBUG: {concept} clean query retrieved mostly: {most_common}")
                
                print(f"{q_type:<20} | {concept:<20} | {f_jaccard:.2f}            | {w_jaccard:.2f}")

    def _jaccard(self, a, b):
        s1 = set(a.cpu().numpy() if isinstance(a, torch.Tensor) else a)
        s2 = set(b.cpu().numpy() if isinstance(b, torch.Tensor) else b)
        if len(s1.union(s2)) == 0: return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

if __name__ == "__main__":
    demo = FlagshipDemo()
    demo.fetch_data()
    demo.embed_and_index()
    demo.run_comparison()
