import pickle
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from semantic_wave_retrieval.engine import WaveRetrievalEngine
from collections import Counter

def jaccard(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if not s1 and not s2: return 0.0
    u = len(s1.union(s2))
    if u == 0: return 0.0
    return len(s1.intersection(s2)) / u

def run_benchmark():
    print("Loading Dataset...")
    try:
        with open("legal_benchmark_data.pkl", "rb") as f:
            corpus, labels, bench_set = pickle.load(f)
    except FileNotFoundError:
        print("Run legal_dataset_gen.py first!")
        return

    print("Embedding Corpus...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus)
    data_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    print("Initializing Engines...")
    # Wave
    wave_engine = WaveRetrievalEngine(data_tensor, k_neighbors=5)
    
    # FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    print("\n--- Running Legal Benchmark ---")
    
    metrics = {
        "FAISS": {"Consistency": [], "DAS": []},
        "Wave": {"Consistency": [], "DAS": []}
    }
    
    def calculate_das(retrieved_indices, labels, criteria):
        """
        Doctrine Alignment Score (DAS)
        1.0 if ALL Required doctrines are present AND NO Forbidden doctrines are present.
        0.0 otherwise.
        """
        retrieved_doctrines = set([labels[i] for i in retrieved_indices if i < len(labels)])
        required = set(criteria['required'])
        forbidden = set(criteria['forbidden'])
        
        # Check coverage
        coverage = required.issubset(retrieved_doctrines)
        # Check safety
        safety = forbidden.isdisjoint(retrieved_doctrines)
        
        return 1.0 if (coverage and safety) else 0.0

    # Iterate Queries
    for entry in bench_set:
        queries = entry['queries']
        das_criteria = entry.get('das_criteria', {})
        clean_q = queries['Clean']
        
        # Get Clean Baseline (Indices) for Consistency
        clean_vec = model.encode([clean_q])[0]
        
        # FAISS Clean
        _, f_clean_idx = index.search(clean_vec.reshape(1, -1), 5)
        f_clean_set = f_clean_idx[0].tolist()
        
        # Wave Clean (Refined)
        w_clean_idx = wave_engine.retrieve_refined_indices(torch.tensor(clean_vec), top_k=5,
                                            wave_params={'T_wave': 8, 'c': 1.0, 'sigma': 1.0}, 
                                            telegrapher_params={'T_damp': 15, 'gamma': 0.8},
                                            poisson_params={'alpha': 0.1})
        w_clean_set = w_clean_idx.cpu().numpy().tolist()
        
        for variant, q_text in queries.items():
            # For DAS, we evaluate ALL variants including Clean.
            # For Consistency, we compare Variant to Clean (skip Clean vs Clean).
            
            q_vec = model.encode([q_text])[0]
            
            # --- FAISS ---
            _, f_idx = index.search(q_vec.reshape(1, -1), 5)
            f_res = f_idx[0].tolist()
            
            # Metric: DAS
            f_das = calculate_das(f_res, labels, das_criteria)
            metrics["FAISS"]["DAS"].append(f_das)
            
            # Metric: Consistency (if not Clean)
            if variant != "Clean":
                f_jac = jaccard(f_clean_set, f_res)
                metrics["FAISS"]["Consistency"].append(f_jac)
            
            # --- WAVE ---
            w_idx = wave_engine.retrieve_refined_indices(torch.tensor(q_vec), top_k=5,
                                          wave_params={'T_wave': 8, 'c': 1.0, 'sigma': 1.0},
                                          telegrapher_params={'T_damp': 15, 'gamma': 0.8},
                                          poisson_params={'alpha': 0.1})
            w_res = w_idx.cpu().numpy().tolist()
            
            # Metric: DAS
            w_das = calculate_das(w_res, labels, das_criteria)
            metrics["Wave"]["DAS"].append(w_das)
            
            # Metric: Consistency
            if variant != "Clean":
                w_jac = jaccard(w_clean_set, w_res)
                metrics["Wave"]["Consistency"].append(w_jac)
            
            # Debug Logic for Complex Queries to see why they might fail
            if entry.get('type') in ['Cross-Doctrine', 'Exception'] and variant == 'Clean':
                print(f"Debug {entry['type']}: {q_text}")
                print(f"  Req: {das_criteria['required']}, Forbid: {das_criteria['forbidden']}")
                print(f"  FAISS Docs: {[labels[i] for i in f_res if i < len(labels)]}")
                print(f"  Wave Docs: {[labels[i] for i in w_res if i < len(labels)]}")
                print(f"  FAISS DAS: {f_das}, Wave DAS: {w_das}")

    # Refusal Benchmark
    print("\n--- Running Refusal Benchmark (OOD) ---")
    ood_queries = [
        "sdlfkjsdflk", 
        "recipe for spicy tacos", 
        "quantum gravity equations",
        "benefits of yoga for cats"
    ]
    
    refusal_scores = []
    
    # We need to compute 'confidence' like in RAG Controller
    # engine.retrieve returns indices. retrieve_basins returns confidence.
    # Let's use retrieve_basins for OOD.
    
    for q in ood_queries:
        q_vec = model.encode([q])[0]
        basins = wave_engine.retrieve_basins(torch.tensor(q_vec), top_k_basins=1,
                                           wave_params={'T_wave': 8, 'c': 1.0, 'sigma': 1.0},
                                           telegrapher_params={'T_damp': 15, 'gamma': 0.8},
                                           poisson_params={'alpha': 0.1})
        if not basins:
            conf = 0.0
        else:
            conf = basins[0]['confidence']
            
        print(f"OOD Query: '{q}' -> Max Conf: {conf:.4f}")
        # Threshold 0.05
        refusal_scores.append(1 if conf < 0.05 else 0)
        
    avg_refusal = np.mean(refusal_scores)
    
    # Calculate CFR (Catastrophic Failure Rate)
    # Failures: 
    # 1. Valid Query: DAS == 0 (Completely wrong doctrine)
    # 2. OOD Query: Refusal == 0 (Hallucination)
    
    # Wave CFR
    w_valid_failures = sum([1 for d in metrics["Wave"]["DAS"] if d < 0.01]) # Strict DAS=0 check
    w_ood_failures = sum([1 for r in refusal_scores if r == 0])
    w_total_queries = len(metrics["Wave"]["DAS"]) + len(ood_queries)
    w_cfr = (w_valid_failures + w_ood_failures) / w_total_queries
    
    # FAISS CFR
    f_valid_failures = sum([1 for d in metrics["FAISS"]["DAS"] if d < 0.01])
    # FAISS OOD Failure is 100% (Refusal = 0 for all)
    f_ood_failures = len(ood_queries) 
    f_total_queries = len(metrics["FAISS"]["DAS"]) + len(ood_queries)
    f_cfr = (f_valid_failures + f_ood_failures) / f_total_queries

    print(f"Refusal Accuracy (Tau=0.05): {avg_refusal:.2f}")

    # Aggregated Results
    print("\n=== Final Results ===")
    print(f"{'Metric':<20} | {'FAISS':<10} | {'Wave':<10}")
    print("-" * 45)
    f_const = np.mean(metrics["FAISS"]["Consistency"])
    w_const = np.mean(metrics["Wave"]["Consistency"])
    print(f"{'Avg Consistency':<20} | {f_const:.4f}     | {w_const:.4f}")
    
    f_das = np.mean(metrics["FAISS"]["DAS"])
    w_das = np.mean(metrics["Wave"]["DAS"])
    print(f"{'Avg DAS (Correctness)':<20} | {f_das:.4f}     | {w_das:.4f}")
    print(f"{'Refusal Accuracy':<20} | {'0.0000':<10} | {avg_refusal:.4f}")
    print(f"{'Catastrophic Fail %':<20} | {f_cfr*100:.1f}%      | {w_cfr*100:.1f}%")

if __name__ == "__main__":
    run_benchmark()
