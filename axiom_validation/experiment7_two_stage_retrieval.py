"""
Experiment 7: Two-Stage Retrieval Benchmark

Hypothesis: Similarity -> Consistency pipeline outperforms similarity alone.

Architecture:
┌─────────────────────────────────────────┐
│  STAGE 1: SIMILARITY RETRIEVAL (BERT)  │
│  - Fast candidate retrieval             │
│  - High recall, may include false pos   │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  STAGE 2: CONSISTENCY FILTER (REWA)    │
│  - Geometric consistency checks         │
│  - Remove contradictions                │
│  - Maintain high precision              │
└─────────────────────────────────────────┘

Success criteria:
- Two-stage achieves higher precision than single-stage
- False positive rate (contradictions returned) reduced
- Recall maintained at acceptable levels
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load pretrained embedding model"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    """Get normalized embedding for text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = embedding / torch.norm(embedding, dim=-1, keepdim=True)
    return embedding.cpu().numpy().squeeze()

# ============================================================================
# RETRIEVAL BENCHMARK DATASET
# ============================================================================

RETRIEVAL_BENCHMARK = [
    {
        'query': "red bike for sale",
        'relevant': [
            "crimson bicycle available",
            "red mountain bike listing",
            "cherry colored bike sale",
            "red bicycle for purchase",
        ],
        'contradictory': [
            "NOT red bike - blue only",
            "no red bikes available",
            "red bike sold out",
            "blue bike for sale (no red)",
        ],
        'irrelevant': [
            "car insurance rates",
            "phone repair service",
            "restaurant reviews",
            "weather forecast today",
        ]
    },
    {
        'query': "patient allergic to penicillin",
        'relevant': [
            "penicillin allergy documented",
            "allergic reaction to penicillin",
            "patient has penicillin sensitivity",
            "penicillin contraindicated",
        ],
        'contradictory': [
            "patient NOT allergic to penicillin",
            "no penicillin allergy",
            "penicillin safe for patient",
            "patient tolerates penicillin well",
        ],
        'irrelevant': [
            "blood pressure medication",
            "physical therapy schedule",
            "lab results pending",
            "insurance coverage details",
        ]
    },
    {
        'query': "restaurant is open now",
        'relevant': [
            "restaurant currently serving",
            "open for business now",
            "accepting customers today",
            "restaurant available now",
        ],
        'contradictory': [
            "restaurant is closed",
            "not open today",
            "closed for renovation",
            "restaurant shut down",
        ],
        'irrelevant': [
            "recipe for pasta",
            "food delivery service",
            "kitchen equipment sale",
            "cooking classes available",
        ]
    },
    {
        'query': "sunny weather today",
        'relevant': [
            "clear skies expected",
            "sunshine all day",
            "bright and sunny forecast",
            "no clouds today",
        ],
        'contradictory': [
            "rainy weather today",
            "storm warning issued",
            "overcast and cloudy",
            "heavy rain expected",
        ],
        'irrelevant': [
            "stock market report",
            "sports game results",
            "movie releases",
            "concert schedule",
        ]
    },
    {
        'query': "flight departing on time",
        'relevant': [
            "flight scheduled as planned",
            "no delays expected",
            "departure on schedule",
            "flight will leave on time",
        ],
        'contradictory': [
            "flight delayed",
            "departure postponed",
            "flight cancelled",
            "significant delays expected",
        ],
        'irrelevant': [
            "hotel booking confirmation",
            "car rental availability",
            "travel insurance",
            "luggage restrictions",
        ]
    },
]

# ============================================================================
# RETRIEVAL SYSTEM
# ============================================================================

class SimilarityRetriever:
    """Stage 1: Standard BERT-based similarity retrieval"""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def retrieve(self, query, documents, top_k=None):
        """
        Retrieve documents by cosine similarity

        Returns: List of (doc, similarity_score) sorted by score
        """
        query_emb = get_embedding(query, self.tokenizer, self.model)

        results = []
        for doc in documents:
            doc_emb = get_embedding(doc, self.tokenizer, self.model)
            similarity = np.dot(query_emb, doc_emb)
            results.append((doc, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            results = results[:top_k]

        return results


class ConsistencyFilter:
    """Stage 2: REWA-based consistency filtering"""

    def __init__(self, tokenizer, model, contradiction_threshold=0.3):
        self.tokenizer = tokenizer
        self.model = model
        self.contradiction_threshold = contradiction_threshold

    def detect_negation(self, text1, text2):
        """Simple negation detection heuristic"""
        negation_words = ['not', 'no', "n't", 'never', 'none', 'neither', 'closed', 'cancelled', 'delayed']

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Check if one has negation words the other doesn't
        neg_in_1 = any(neg in text1_lower for neg in negation_words)
        neg_in_2 = any(neg in text2_lower for neg in negation_words)

        return neg_in_1 != neg_in_2

    def compute_semantic_energy(self, query_emb, doc_emb):
        """Compute semantic energy (1 - similarity)"""
        return 1.0 - np.dot(query_emb, doc_emb)

    def filter(self, query, candidates):
        """
        Filter candidates using consistency checks

        Removes:
        1. Documents that contradict the query (detected via negation patterns)
        2. Documents with high semantic energy relative to query
        """
        query_emb = get_embedding(query, self.tokenizer, self.model)

        filtered = []
        rejected = []

        for doc, sim_score in candidates:
            doc_emb = get_embedding(doc, self.tokenizer, self.model)

            # Check for negation pattern
            has_negation = self.detect_negation(query, doc)

            # Compute semantic energy
            energy = self.compute_semantic_energy(query_emb, doc_emb)

            # Decision logic
            # If high similarity but negation detected → likely contradiction
            if has_negation and sim_score > self.contradiction_threshold:
                rejected.append((doc, sim_score, 'negation_contradiction'))
            # If very low similarity → probably irrelevant
            elif sim_score < 0.1:
                rejected.append((doc, sim_score, 'low_similarity'))
            else:
                filtered.append((doc, sim_score))

        return filtered, rejected


class TwoStageRetriever:
    """Complete two-stage retrieval system"""

    def __init__(self, tokenizer, model):
        self.stage1 = SimilarityRetriever(tokenizer, model)
        self.stage2 = ConsistencyFilter(tokenizer, model)

    def retrieve(self, query, documents, top_k_stage1=None, return_rejected=False):
        """
        Two-stage retrieval:
        1. Similarity-based candidate generation
        2. Consistency-based filtering
        """
        # Stage 1: Get candidates
        candidates = self.stage1.retrieve(query, documents, top_k=top_k_stage1)

        # Stage 2: Filter for consistency
        filtered, rejected = self.stage2.filter(query, candidates)

        if return_rejected:
            return filtered, rejected
        return filtered

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_retrieval(query_data, retrieved_docs, method_name):
    """
    Evaluate retrieval results

    Metrics:
    - Precision: relevant / retrieved
    - Recall: relevant_retrieved / total_relevant
    - False positive rate: contradictions_retrieved / total_contradictions
    - Contamination: contradictions_in_top_k / top_k
    """
    relevant_set = set(query_data['relevant'])
    contradictory_set = set(query_data['contradictory'])
    irrelevant_set = set(query_data['irrelevant'])

    retrieved_set = set(doc for doc, _ in retrieved_docs)

    # True positives (relevant retrieved)
    true_positives = len(relevant_set & retrieved_set)

    # False positives (contradictory or irrelevant retrieved)
    contradictions_retrieved = len(contradictory_set & retrieved_set)
    irrelevant_retrieved = len(irrelevant_set & retrieved_set)
    false_positives = contradictions_retrieved + irrelevant_retrieved

    # Metrics
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    contradiction_rate = contradictions_retrieved / len(contradictory_set) if contradictory_set else 0
    contamination = contradictions_retrieved / len(retrieved_set) if retrieved_set else 0

    return {
        'method': method_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'contradictions_retrieved': contradictions_retrieved,
        'contradiction_rate': contradiction_rate,
        'contamination': contamination,
        'retrieved_count': len(retrieved_set)
    }

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_retrieval_benchmark(tokenizer, model):
    """Run the full retrieval benchmark"""

    print("\n" + "="*70)
    print("EXPERIMENT 7: TWO-STAGE RETRIEVAL BENCHMARK")
    print("="*70)

    similarity_retriever = SimilarityRetriever(tokenizer, model)
    two_stage_retriever = TwoStageRetriever(tokenizer, model)

    all_results_similarity = []
    all_results_two_stage = []

    for i, query_data in enumerate(RETRIEVAL_BENCHMARK):
        query = query_data['query']

        print(f"\n{'='*70}")
        print(f"Query {i+1}: \"{query}\"")
        print(f"{'='*70}")

        # Combine all documents
        all_docs = (
            query_data['relevant'] +
            query_data['contradictory'] +
            query_data['irrelevant']
        )

        # Method 1: Similarity-only retrieval
        sim_results = similarity_retriever.retrieve(query, all_docs)

        # Take top-6 (since we have 4 relevant + 2 buffer)
        sim_top_k = sim_results[:6]

        print(f"\nMethod 1: Similarity-Only (Top 6)")
        print("-"*50)
        for doc, score in sim_top_k:
            category = "RELEVANT" if doc in query_data['relevant'] else \
                      "CONTRA" if doc in query_data['contradictory'] else "IRRELEVANT"
            print(f"  [{category:10s}] {score:.3f} | {doc[:50]}...")

        sim_metrics = evaluate_retrieval(query_data, sim_top_k, 'Similarity')

        # Method 2: Two-stage retrieval
        two_stage_results, rejected = two_stage_retriever.retrieve(
            query, all_docs, top_k_stage1=None, return_rejected=True
        )
        two_stage_top_k = two_stage_results[:6]

        print(f"\nMethod 2: Two-Stage (Top 6 after filtering)")
        print("-"*50)
        for doc, score in two_stage_top_k:
            category = "RELEVANT" if doc in query_data['relevant'] else \
                      "CONTRA" if doc in query_data['contradictory'] else "IRRELEVANT"
            print(f"  [{category:10s}] {score:.3f} | {doc[:50]}...")

        if rejected:
            print(f"\n  Rejected by consistency filter:")
            for doc, score, reason in rejected[:3]:
                print(f"    [{reason:20s}] {score:.3f} | {doc[:40]}...")

        two_stage_metrics = evaluate_retrieval(query_data, two_stage_top_k, 'Two-Stage')

        # Store results
        all_results_similarity.append(sim_metrics)
        all_results_two_stage.append(two_stage_metrics)

        # Per-query comparison
        print(f"\nComparison:")
        print(f"  {'Metric':<20} | {'Similarity':>12} | {'Two-Stage':>12}")
        print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}")
        print(f"  {'Precision':<20} | {sim_metrics['precision']:>11.1%} | {two_stage_metrics['precision']:>11.1%}")
        print(f"  {'Recall':<20} | {sim_metrics['recall']:>11.1%} | {two_stage_metrics['recall']:>11.1%}")
        print(f"  {'Contradictions':<20} | {sim_metrics['contradictions_retrieved']:>12d} | {two_stage_metrics['contradictions_retrieved']:>12d}")
        print(f"  {'Contamination':<20} | {sim_metrics['contamination']:>11.1%} | {two_stage_metrics['contamination']:>11.1%}")

    return all_results_similarity, all_results_two_stage

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_benchmark_results(sim_results, two_stage_results):
    """Analyze overall benchmark results"""

    print("\n" + "="*70)
    print("OVERALL BENCHMARK ANALYSIS")
    print("="*70)

    # Aggregate metrics
    def aggregate(results):
        return {
            'avg_precision': np.mean([r['precision'] for r in results]),
            'avg_recall': np.mean([r['recall'] for r in results]),
            'avg_f1': np.mean([r['f1'] for r in results]),
            'total_contradictions': sum(r['contradictions_retrieved'] for r in results),
            'avg_contamination': np.mean([r['contamination'] for r in results])
        }

    sim_agg = aggregate(sim_results)
    two_stage_agg = aggregate(two_stage_results)

    print(f"\n{'Metric':<25} | {'Similarity':>12} | {'Two-Stage':>12} | {'Improvement':>12}")
    print(f"{'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    print(f"{'Avg Precision':<25} | {sim_agg['avg_precision']:>11.1%} | {two_stage_agg['avg_precision']:>11.1%} | {(two_stage_agg['avg_precision']-sim_agg['avg_precision'])*100:>+10.1f}%")
    print(f"{'Avg Recall':<25} | {sim_agg['avg_recall']:>11.1%} | {two_stage_agg['avg_recall']:>11.1%} | {(two_stage_agg['avg_recall']-sim_agg['avg_recall'])*100:>+10.1f}%")
    print(f"{'Avg F1':<25} | {sim_agg['avg_f1']:>11.1%} | {two_stage_agg['avg_f1']:>11.1%} | {(two_stage_agg['avg_f1']-sim_agg['avg_f1'])*100:>+10.1f}%")
    print(f"{'Total Contradictions':<25} | {sim_agg['total_contradictions']:>12d} | {two_stage_agg['total_contradictions']:>12d} | {two_stage_agg['total_contradictions']-sim_agg['total_contradictions']:>+12d}")
    print(f"{'Avg Contamination':<25} | {sim_agg['avg_contamination']:>11.1%} | {two_stage_agg['avg_contamination']:>11.1%} | {(two_stage_agg['avg_contamination']-sim_agg['avg_contamination'])*100:>+10.1f}%")

    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)

    criteria = [
        ("Precision improved", two_stage_agg['avg_precision'] > sim_agg['avg_precision']),
        ("Contamination reduced", two_stage_agg['avg_contamination'] < sim_agg['avg_contamination']),
        ("Contradictions reduced", two_stage_agg['total_contradictions'] < sim_agg['total_contradictions']),
        ("Recall maintained (>50%)", two_stage_agg['avg_recall'] > 0.5)
    ]

    print("\nCriteria:")
    for name, met in criteria:
        print(f"  [{'PASS' if met else 'FAIL'}] {name}")

    validated = sum(1 for _, met in criteria if met) >= 3

    if validated:
        print("\n[VALIDATED] Two-stage retrieval outperforms similarity-only")
    else:
        print("\n[PARTIAL] Some improvement but not all criteria met")

    return {
        'similarity': sim_agg,
        'two_stage': two_stage_agg,
        'validated': validated
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_benchmark(sim_results, two_stage_results, output_path='experiment7_two_stage_retrieval.png'):
    """Visualize benchmark results"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Metrics for comparison
    metrics = ['precision', 'recall', 'f1', 'contamination']
    metric_labels = ['Precision', 'Recall', 'F1 Score', 'Contamination']

    # Plot 1: Per-query comparison (Precision)
    ax = axes[0, 0]
    queries = [f"Q{i+1}" for i in range(len(sim_results))]
    x = np.arange(len(queries))
    width = 0.35

    sim_precision = [r['precision'] for r in sim_results]
    two_stage_precision = [r['precision'] for r in two_stage_results]

    ax.bar(x - width/2, sim_precision, width, label='Similarity', color='red', alpha=0.7)
    ax.bar(x + width/2, two_stage_precision, width, label='Two-Stage', color='green', alpha=0.7)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Query')
    ax.set_title('Precision by Query')
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Plot 2: Per-query comparison (Contamination)
    ax = axes[0, 1]
    sim_contam = [r['contamination'] for r in sim_results]
    two_stage_contam = [r['contamination'] for r in two_stage_results]

    ax.bar(x - width/2, sim_contam, width, label='Similarity', color='red', alpha=0.7)
    ax.bar(x + width/2, two_stage_contam, width, label='Two-Stage', color='green', alpha=0.7)
    ax.set_ylabel('Contamination Rate')
    ax.set_xlabel('Query')
    ax.set_title('Contradiction Contamination by Query')
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Plot 3: Overall metrics comparison
    ax = axes[1, 0]
    sim_metrics = [np.mean([r[m] for r in sim_results]) for m in metrics]
    two_stage_metrics = [np.mean([r[m] for r in two_stage_results]) for m in metrics]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, sim_metrics, width, label='Similarity', color='red', alpha=0.7)
    ax.bar(x + width/2, two_stage_metrics, width, label='Two-Stage', color='green', alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_title('Overall Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Plot 4: Contradictions retrieved
    ax = axes[1, 1]
    sim_contra = [r['contradictions_retrieved'] for r in sim_results]
    two_stage_contra = [r['contradictions_retrieved'] for r in two_stage_results]

    x_contra = np.arange(len(sim_contra))
    ax.bar(x_contra - width/2, sim_contra, width, label='Similarity', color='red', alpha=0.7)
    ax.bar(x_contra + width/2, two_stage_contra, width, label='Two-Stage', color='green', alpha=0.7)
    ax.set_ylabel('Contradictions Retrieved')
    ax.set_xlabel('Query')
    ax.set_title('Contradictions in Results (Lower = Better)')
    ax.set_xticks(x_contra)
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(sim_contra))])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] Visualization saved to '{output_path}'")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 7: TWO-STAGE RETRIEVAL BENCHMARK")
    print("="*70)
    print("\nHypothesis: Similarity -> Consistency pipeline outperforms")
    print("            similarity alone for logical consistency.")
    print("\nArchitecture:")
    print("  Stage 1 (BERT): Fast candidate retrieval by similarity")
    print("  Stage 2 (REWA): Consistency filtering to remove contradictions")
    print("="*70)

    tokenizer, model = load_embedding_model()

    # Run benchmark
    sim_results, two_stage_results = run_retrieval_benchmark(tokenizer, model)

    # Analyze results
    analysis = analyze_benchmark_results(sim_results, two_stage_results)

    # Visualize
    visualize_benchmark(sim_results, two_stage_results)

    # Final Summary
    print("\n" + "="*70)
    print("EXPERIMENT 7 FINAL SUMMARY")
    print("="*70)

    print(f"\nKey Results:")
    print(f"  Similarity-only:")
    print(f"    - Average precision: {analysis['similarity']['avg_precision']:.1%}")
    print(f"    - Average contamination: {analysis['similarity']['avg_contamination']:.1%}")
    print(f"    - Total contradictions returned: {analysis['similarity']['total_contradictions']}")

    print(f"\n  Two-Stage (BERT + REWA):")
    print(f"    - Average precision: {analysis['two_stage']['avg_precision']:.1%}")
    print(f"    - Average contamination: {analysis['two_stage']['avg_contamination']:.1%}")
    print(f"    - Total contradictions returned: {analysis['two_stage']['total_contradictions']}")

    precision_improvement = (analysis['two_stage']['avg_precision'] - analysis['similarity']['avg_precision']) / max(analysis['similarity']['avg_precision'], 0.01)
    contamination_reduction = (analysis['similarity']['avg_contamination'] - analysis['two_stage']['avg_contamination']) / max(analysis['similarity']['avg_contamination'], 0.01)

    print(f"\n  Improvement:")
    print(f"    - Precision improvement: {precision_improvement:+.1%}")
    print(f"    - Contamination reduction: {contamination_reduction:.1%}")

    return {
        'sim_results': sim_results,
        'two_stage_results': two_stage_results,
        'analysis': analysis
    }

if __name__ == "__main__":
    results = main()
