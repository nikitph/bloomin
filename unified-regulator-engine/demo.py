#!/usr/bin/env python3
"""
Unified Regulator Engine - Demonstration
=========================================

This script demonstrates the three modes of the URE:
1. Retrieval: Finding relevant items for a query
2. Clustering: Discovering natural groupings
3. Decision: Selecting among candidates

All three are just different configurations of the same dynamical system.

Key insight:
    - Same loop
    - Different operators
    - Unified confidence semantics
"""

import numpy as np
import time
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from ure_core import (
    UnifiedRegulatorEngine,
    RegulatorParams,
    Mode,
    quick_retrieve,
    quick_cluster,
    quick_decide,
)


def generate_clustered_data(
    n_clusters: int = 5,
    points_per_cluster: int = 100,
    dim: int = 64,
    noise_scale: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic clustered data."""
    np.random.seed(seed)

    # Generate cluster centers on unit sphere
    centers = np.random.randn(n_clusters, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # Generate points around centers
    points = []
    labels = []

    for i, center in enumerate(centers):
        noise = np.random.randn(points_per_cluster, dim) * noise_scale
        cluster_points = center + noise
        cluster_points = cluster_points / np.linalg.norm(cluster_points, axis=1, keepdims=True)
        points.append(cluster_points)
        labels.extend([i] * points_per_cluster)

    return np.vstack(points).astype(np.float32), np.array(labels)


def demo_retrieval():
    """Demonstrate retrieval mode."""
    print("\n" + "=" * 70)
    print("DEMO 1: RETRIEVAL MODE")
    print("=" * 70)
    print("\nPhysics: Wave → Telegrapher → Poisson basins")
    print("Use case: Find relevant documents/items for a query\n")

    # Generate data
    corpus, labels = generate_clustered_data(
        n_clusters=10,
        points_per_cluster=100,
        dim=128
    )
    print(f"Corpus: {corpus.shape[0]} items, {corpus.shape[1]} dimensions")

    # Create a query (pick a point and add noise)
    query_idx = 42
    query = corpus[query_idx] + np.random.randn(128) * 0.1
    query = query / np.linalg.norm(query)
    true_cluster = labels[query_idx]
    print(f"Query from cluster {true_cluster}")

    # Build engine and retrieve
    engine = UnifiedRegulatorEngine(
        params=RegulatorParams(T_explore=30, T_select=30, tau=0.2)
    )
    engine.build_index(corpus, k=15)

    start = time.time()
    result = engine.retrieve(query, k=10)
    elapsed = time.time() - start

    print(f"\nResults (time: {elapsed*1000:.1f}ms):")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Refused: {result.refused}")
    print(f"  Retrieved {len(result.output)} items")

    if not result.refused:
        retrieved_labels = labels[result.output]
        precision = np.mean(retrieved_labels == true_cluster)
        print(f"  Precision (same cluster): {precision*100:.1f}%")

    # Test with noise query (should have low confidence)
    print("\nTesting with random noise query...")
    noise_query = np.random.randn(128)
    noise_query = noise_query / np.linalg.norm(noise_query)
    noise_result = engine.retrieve(noise_query, k=10)
    print(f"  Confidence: {noise_result.confidence:.3f}")
    print(f"  Refused: {noise_result.refused} (expected: True)")


def demo_clustering():
    """Demonstrate clustering mode."""
    print("\n" + "=" * 70)
    print("DEMO 2: CLUSTERING MODE")
    print("=" * 70)
    print("\nPhysics: Fokker-Planck → Cahn-Hilliard → Phase domains")
    print("Use case: Discover natural groupings in data\n")

    # Generate data with clear clusters
    data, true_labels = generate_clustered_data(
        n_clusters=5,
        points_per_cluster=50,
        dim=32,
        noise_scale=0.15
    )
    print(f"Data: {data.shape[0]} points, {data.shape[1]} dimensions")
    print(f"True clusters: {len(np.unique(true_labels))}")

    # Run clustering
    engine = UnifiedRegulatorEngine(
        params=RegulatorParams(
            T_explore=50,
            T_select=100,
            epsilon=0.05,
            tau=0.15
        )
    )

    start = time.time()
    result = engine.cluster(data)
    elapsed = time.time() - start

    print(f"\nResults (time: {elapsed*1000:.1f}ms):")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Refused: {result.refused}")
    print(f"  Found {result.metadata['n_clusters']} clusters")

    if not result.refused:
        # Compute clustering quality (adjusted rand index)
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        pred_labels = result.output
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        print(f"  Adjusted Rand Index: {ari:.3f}")
        print(f"  Normalized MI: {nmi:.3f}")


def demo_decision():
    """Demonstrate decision mode."""
    print("\n" + "=" * 70)
    print("DEMO 3: DECISION MODE")
    print("=" * 70)
    print("\nPhysics: Schr\u00f6dinger → Fisher-KPP → Winner")
    print("Use case: Select best option from candidates\n")

    # Create candidates with different quality levels
    np.random.seed(42)
    n_candidates = 20
    dim = 32

    # Generate candidates
    candidates = np.random.randn(n_candidates, dim).astype(np.float32)
    candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

    # Define a "target" that we want to get close to
    target = np.random.randn(dim)
    target = target / np.linalg.norm(target)

    # Compute ground truth quality (similarity to target)
    true_quality = candidates @ target
    best_candidate = np.argmax(true_quality)

    print(f"Candidates: {n_candidates}")
    print(f"Ground truth best: candidate {best_candidate} (quality: {true_quality[best_candidate]:.3f})")

    # Use quality as potential (low = good, so invert)
    V_loss = 1 - (true_quality - true_quality.min()) / (true_quality.max() - true_quality.min() + 1e-10)

    # Run decision
    engine = UnifiedRegulatorEngine(
        params=RegulatorParams(
            T_explore=40,
            T_select=60,
            tau=0.25
        )
    )

    start = time.time()
    result = engine.decide(candidates, V_loss=V_loss)
    elapsed = time.time() - start

    print(f"\nResults (time: {elapsed*1000:.1f}ms):")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Refused: {result.refused}")
    print(f"  Selected: candidate {result.output}")
    print(f"  Selected quality: {true_quality[result.output]:.3f}")
    print(f"  Correct: {result.output == best_candidate}")

    # Show top-k
    print(f"\n  Top 5 by activation: {result.metadata['top_k'][:5]}")
    print(f"  Top 5 by true quality: {np.argsort(true_quality)[-5:][::-1].tolist()}")

    # Test with uniform quality (should refuse or have low confidence)
    print("\nTesting with uniform quality (ambiguous)...")
    uniform_loss = np.ones(n_candidates) * 0.5
    ambiguous_result = engine.decide(candidates, V_loss=uniform_loss)
    print(f"  Confidence: {ambiguous_result.confidence:.3f}")


def demo_cascade():
    """Demonstrate cascaded modes."""
    print("\n" + "=" * 70)
    print("DEMO 4: CASCADED MODES")
    print("=" * 70)
    print("\nPhysics: Retrieval → Decision (coarse → fine)")
    print("Use case: Find candidates then pick best\n")

    # Large corpus
    corpus, labels = generate_clustered_data(
        n_clusters=20,
        points_per_cluster=200,
        dim=64
    )
    print(f"Corpus: {corpus.shape[0]} items")

    # Query
    query_idx = 42
    query = corpus[query_idx] + np.random.randn(64) * 0.1
    query = query / np.linalg.norm(query)

    # Single-mode retrieval
    engine = UnifiedRegulatorEngine()
    engine.build_index(corpus, k=15)

    print("\nSingle-mode retrieval:")
    start = time.time()
    single_result = engine.retrieve(query, k=50)
    single_time = time.time() - start
    print(f"  Time: {single_time*1000:.1f}ms")
    print(f"  Confidence: {single_result.confidence:.3f}")

    # Cascaded: Retrieval → Decision
    print("\nCascaded retrieval → decision:")
    start = time.time()
    results = engine.cascade(query, [Mode.RETRIEVAL, Mode.DECISION])
    cascade_time = time.time() - start
    print(f"  Time: {cascade_time*1000:.1f}ms")
    print(f"  Stage 1 (Retrieval) confidence: {results[0].confidence:.3f}")
    if len(results) > 1:
        print(f"  Stage 2 (Decision) confidence: {results[1].confidence:.3f}")
        print(f"  Final selection: {results[1].output}")


def demo_refusal():
    """Demonstrate principled refusal."""
    print("\n" + "=" * 70)
    print("DEMO 5: PRINCIPLED REFUSAL")
    print("=" * 70)
    print("\nKey insight: if max(confidence) < tau, system refuses")
    print("This handles: ambiguous queries, noise, adversarial inputs\n")

    # Create corpus
    corpus, _ = generate_clustered_data(n_clusters=5, points_per_cluster=100, dim=64)

    engine = UnifiedRegulatorEngine(
        params=RegulatorParams(tau=0.4)  # Higher threshold
    )
    engine.build_index(corpus, k=10)

    # Test 1: Valid query (should accept)
    valid_query = corpus[0] + np.random.randn(64) * 0.05
    valid_query = valid_query / np.linalg.norm(valid_query)
    valid_result = engine.retrieve(valid_query, k=10)
    print(f"Valid query:   confidence={valid_result.confidence:.3f}, refused={valid_result.refused}")

    # Test 2: Pure noise (should refuse)
    noise_query = np.random.randn(64)
    noise_query = noise_query / np.linalg.norm(noise_query)
    noise_result = engine.retrieve(noise_query, k=10)
    print(f"Noise query:   confidence={noise_result.confidence:.3f}, refused={noise_result.refused}")

    # Test 3: Adversarial (equidistant from everything)
    adversarial = np.mean(corpus, axis=0)
    adversarial = adversarial / np.linalg.norm(adversarial)
    adv_result = engine.retrieve(adversarial, k=10)
    print(f"Adversarial:   confidence={adv_result.confidence:.3f}, refused={adv_result.refused}")

    print("\nFAISS cannot do this. URE does this by design.")


def demo_comparison_faiss():
    """Compare with FAISS baseline."""
    print("\n" + "=" * 70)
    print("DEMO 6: COMPARISON WITH FAISS")
    print("=" * 70)

    try:
        import faiss
    except ImportError:
        print("\nFAISS not installed. Skipping comparison.")
        return

    # Generate data
    corpus, labels = generate_clustered_data(
        n_clusters=10,
        points_per_cluster=1000,
        dim=128
    )
    queries = corpus[::100].copy()  # 100 queries
    print(f"\nCorpus: {corpus.shape[0]}, Queries: {len(queries)}")

    # FAISS
    print("\nFAISS (Flat L2):")
    faiss_index = faiss.IndexFlatL2(128)
    faiss_index.add(corpus)

    start = time.time()
    D, I = faiss_index.search(queries, 10)
    faiss_time = time.time() - start
    print(f"  Time: {faiss_time*1000:.1f}ms ({faiss_time/len(queries)*1000:.3f}ms/query)")

    # URE
    print("\nURE (Retrieval mode):")
    engine = UnifiedRegulatorEngine(
        params=RegulatorParams(T_explore=20, T_select=20)
    )
    engine.build_index(corpus, k=15)

    ure_times = []
    ure_results = []
    for q in queries:
        start = time.time()
        result = engine.retrieve(q, k=10)
        ure_times.append(time.time() - start)
        ure_results.append(result)

    avg_time = np.mean(ure_times)
    avg_confidence = np.mean([r.confidence for r in ure_results])
    n_refused = sum(r.refused for r in ure_results)

    print(f"  Time: {sum(ure_times)*1000:.1f}ms ({avg_time*1000:.3f}ms/query)")
    print(f"  Avg confidence: {avg_confidence:.3f}")
    print(f"  Refused: {n_refused}/{len(queries)}")

    # Recall comparison
    faiss_results = [set(row.tolist()) for row in I]

    recall_sum = 0
    valid_queries = 0
    for i, result in enumerate(ure_results):
        if not result.refused and len(result.output) > 0:
            ure_set = set(result.output[:10])
            faiss_set = faiss_results[i]
            overlap = len(ure_set & faiss_set)
            recall_sum += overlap / 10
            valid_queries += 1

    if valid_queries > 0:
        avg_recall = recall_sum / valid_queries
        print(f"  Overlap with FAISS: {avg_recall*100:.1f}%")

    print(f"\n  Speedup (FAISS/URE): {avg_time*1000/faiss_time*len(queries)*1000:.2f}x slower")
    print("  But: URE provides confidence + refusal + interpretability")


def main():
    """Run all demos."""
    print("=" * 70)
    print("UNIFIED REGULATOR ENGINE - DEMONSTRATION")
    print("=" * 70)
    print("\n'Computation is controlled energy flow on a manifold,")
    print(" regulated by dissipation.'")
    print("\nThis engine unifies retrieval, clustering, and decision-making")
    print("as different phases of the same physics.\n")

    demo_retrieval()
    demo_clustering()
    demo_decision()
    demo_cascade()
    demo_refusal()
    demo_comparison_faiss()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
What you've seen:

1. RETRIEVAL: Wave → Telegrapher → Poisson basins
   - Query-driven exploration
   - Energy dissipation to stable attractors
   - Basin mass as confidence

2. CLUSTERING: Fokker-Planck → Cahn-Hilliard → Phase domains
   - Drift-diffusion exploration
   - Phase separation selection
   - Domain mass as confidence

3. DECISION: Schr\u00f6dinger → Fisher-KPP → Winner
   - Loss-guided exploration
   - Winner-take-all selection
   - Activation mass as confidence

4. CASCADE: Coarse → Fine
   - Chain modes for multi-stage reasoning

5. REFUSAL: if max(confidence) < tau: REFUSE
   - Principled uncertainty handling
   - No heuristics, just physics

This is not an algorithm. It's a computational physics substrate.
""")


if __name__ == "__main__":
    main()
