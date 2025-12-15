#!/usr/bin/env python3
"""
Two-Stage Retrieval Benchmark
=============================
The correct benchmark: Thermal Bloom as a COARSE retriever, not end-to-end.

Architecture:
  Stage 1: Fast coarse search in 2D projected space (O(k) gradient descent)
  Stage 2: Exact reranking in original space (O(k) distance computations)

Success criterion: ≥90% recall vs FAISS, ≥5x faster Stage 1
"""

import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# THERMAL BLOOM (Optimized for coarse retrieval)
# ============================================================================

class ThermalBloomCoarse:
    """Thermal Bloom optimized for fast coarse candidate retrieval"""

    def __init__(self, grid_size: int = 256, sigma: float = 1.0):
        self.grid_size = grid_size
        self.sigma = sigma
        self.grid = None
        self.items = {}  # (x, y) -> list of item indices
        self.points_2d = None
        self.pca = None
        self.data_min = None
        self.data_max = None

    def build(self, vectors: np.ndarray):
        """Build index with automatic PCA projection"""
        # PCA to 2D
        self.pca = PCA(n_components=2)
        projected = self.pca.fit_transform(vectors)

        # Normalize to grid range
        self.data_min = projected.min(axis=0)
        self.data_max = projected.max(axis=0)
        margin = (self.data_max - self.data_min) * 0.05
        self.data_min -= margin
        self.data_max += margin

        self.points_2d = self._normalize(projected)

        # Build grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for i, pt in enumerate(self.points_2d):
            x, y = self._to_grid(pt)
            self.grid[x, y] = 1.0
            if (x, y) not in self.items:
                self.items[(x, y)] = []
            self.items[(x, y)].append(i)

        # Thermal diffusion
        self.grid = gaussian_filter(self.grid, sigma=self.sigma)

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to [0, 1] range"""
        return (points - self.data_min) / (self.data_max - self.data_min + 1e-8)

    def _to_grid(self, pt: np.ndarray) -> Tuple[int, int]:
        """Convert normalized point to grid coordinates"""
        x = int(np.clip(pt[0] * (self.grid_size - 1), 0, self.grid_size - 1))
        y = int(np.clip(pt[1] * (self.grid_size - 1), 0, self.grid_size - 1))
        return x, y

    def query_coarse(self, query_vector: np.ndarray, k: int = 100,
                     max_steps: int = 30, search_radius: int = 5) -> Tuple[List[int], float]:
        """
        Fast coarse retrieval via gradient ascent.
        Returns: (candidate_indices, query_time)
        """
        start = time.perf_counter()

        # Project query to 2D
        query_2d = self.pca.transform(query_vector.reshape(1, -1))[0]
        query_2d = self._normalize(query_2d.reshape(1, -1))[0]

        x, y = self._to_grid(query_2d)
        visited = [(x, y)]

        # Gradient ascent
        for _ in range(max_steps):
            if x <= 0 or x >= self.grid_size - 1 or y <= 0 or y >= self.grid_size - 1:
                break

            dx = (self.grid[x + 1, y] - self.grid[x - 1, y]) / 2
            dy = (self.grid[x, y + 1] - self.grid[x, y - 1]) / 2

            if abs(dx) < 1e-7 and abs(dy) < 1e-7:
                break

            new_x = x + (1 if dx > 0.0005 else (-1 if dx < -0.0005 else 0))
            new_y = y + (1 if dy > 0.0005 else (-1 if dy < -0.0005 else 0))

            if new_x == x and new_y == y:
                break

            x, y = new_x, new_y
            visited.append((x, y))

        # Collect candidates from neighborhood
        candidates = set()
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = np.clip(x + dx, 0, self.grid_size - 1)
                ny = np.clip(y + dy, 0, self.grid_size - 1)
                if (nx, ny) in self.items:
                    candidates.update(self.items[(nx, ny)])

        # Also collect from path
        for vx, vy in visited:
            if (vx, vy) in self.items:
                candidates.update(self.items[(vx, vy)])

        elapsed = time.perf_counter() - start
        return list(candidates)[:k], elapsed


# ============================================================================
# RERANKER (Stage 2)
# ============================================================================

def rerank_exact(query_vector: np.ndarray, candidate_indices: List[int],
                 index_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, float]:
    """
    Exact reranking of candidates using full-dimensional distance.
    Returns: (top_k_indices, rerank_time)
    """
    start = time.perf_counter()

    if not candidate_indices:
        return np.array([-1] * k), time.perf_counter() - start

    # Compute exact distances
    candidate_vectors = index_vectors[candidate_indices]
    distances = np.sum((candidate_vectors - query_vector) ** 2, axis=1)

    # Sort and return top-k
    sorted_idx = np.argsort(distances)[:k]
    result = [candidate_indices[i] for i in sorted_idx]

    # Pad if needed
    while len(result) < k:
        result.append(-1)

    elapsed = time.perf_counter() - start
    return np.array(result), elapsed


# ============================================================================
# FAISS BASELINE (Two-stage for fair comparison)
# ============================================================================

def faiss_coarse_search(query_vector: np.ndarray, faiss_index,
                        k: int = 100) -> Tuple[List[int], float]:
    """FAISS coarse search"""
    start = time.perf_counter()
    D, I = faiss_index.search(query_vector.reshape(1, -1).astype('float32'), k)
    elapsed = time.perf_counter() - start
    return I[0].tolist(), elapsed


# ============================================================================
# BENCHMARK
# ============================================================================

@dataclass
class TwoStageResult:
    method: str
    recall_at_1: float
    recall_at_10: float
    stage1_time_ms: float
    stage2_time_ms: float
    total_time_ms: float
    avg_candidates: float
    n_queries: int
    config: dict


def compute_recall(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k"""
    hits = 0
    for pred, gt in zip(predictions, ground_truth):
        pred_set = set(p for p in pred[:k] if p >= 0)
        gt_set = set(gt[:k].tolist())
        if pred_set & gt_set:
            hits += 1
    return hits / len(predictions)


def run_two_stage_benchmark(n_index: int = 100000, n_query: int = 10000,
                            dim: int = 384, n_clusters: int = 100):
    """Run the definitive two-stage benchmark"""
    print("=" * 70)
    print("TWO-STAGE RETRIEVAL BENCHMARK")
    print("Thermal Bloom (coarse) + Exact Rerank vs FAISS (coarse) + Exact Rerank")
    print("=" * 70)

    # Generate synthetic embeddings with cluster structure
    print(f"\nGenerating {n_index + n_query} embeddings ({dim}D, {n_clusters} clusters)...")
    np.random.seed(42)

    # Cluster centers
    centers = np.random.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Generate samples
    embeddings = []
    for i in range(n_index + n_query):
        center = centers[i % n_clusters]
        noise = np.random.randn(dim) * 0.25
        sample = center + noise
        sample /= np.linalg.norm(sample)
        embeddings.append(sample)
    embeddings = np.array(embeddings, dtype=np.float32)

    index_vectors = embeddings[:n_index]
    query_vectors = embeddings[n_index:]

    print(f"Index: {n_index}, Query: {n_query}, Dim: {dim}")

    # Compute ground truth
    print("\nComputing ground truth (brute force k-NN)...")
    nn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
    nn.fit(index_vectors)
    _, ground_truth = nn.kneighbors(query_vectors)

    results = []

    # =========================================================================
    # THERMAL BLOOM TWO-STAGE
    # =========================================================================
    print("\n" + "-" * 70)
    print("THERMAL BLOOM TWO-STAGE")
    print("-" * 70)

    configs = [
        (256, 0.5, 3, 50),   # grid_size, sigma, search_radius, k_coarse
        (256, 1.0, 5, 100),
        (512, 0.5, 3, 50),
        (512, 1.0, 5, 100),
        (256, 1.0, 5, 200),  # More candidates
    ]

    for grid_size, sigma, search_radius, k_coarse in configs:
        print(f"\n  Config: grid={grid_size}, σ={sigma}, r={search_radius}, k_coarse={k_coarse}")

        # Build
        build_start = time.time()
        thermal = ThermalBloomCoarse(grid_size=grid_size, sigma=sigma)
        thermal.build(index_vectors)
        build_time = time.time() - build_start
        print(f"    Build time: {build_time:.2f}s")

        # Query
        all_predictions = []
        total_stage1_time = 0
        total_stage2_time = 0
        total_candidates = 0

        for query in query_vectors:
            # Stage 1: Coarse
            candidates, s1_time = thermal.query_coarse(query, k=k_coarse, search_radius=search_radius)
            total_stage1_time += s1_time
            total_candidates += len(candidates)

            # Stage 2: Rerank
            final, s2_time = rerank_exact(query, candidates, index_vectors, k=10)
            total_stage2_time += s2_time

            all_predictions.append(final)

        all_predictions = np.array(all_predictions)

        recall_1 = compute_recall(all_predictions, ground_truth, 1)
        recall_10 = compute_recall(all_predictions, ground_truth, 10)

        result = TwoStageResult(
            method=f"Thermal_g{grid_size}_s{sigma}_r{search_radius}_k{k_coarse}",
            recall_at_1=recall_1,
            recall_at_10=recall_10,
            stage1_time_ms=total_stage1_time * 1000 / n_query,
            stage2_time_ms=total_stage2_time * 1000 / n_query,
            total_time_ms=(total_stage1_time + total_stage2_time) * 1000 / n_query,
            avg_candidates=total_candidates / n_query,
            n_queries=n_query,
            config={"grid_size": grid_size, "sigma": sigma, "search_radius": search_radius, "k_coarse": k_coarse}
        )
        results.append(result)

        print(f"    Recall@1: {recall_1*100:.1f}%, Recall@10: {recall_10*100:.1f}%")
        print(f"    Stage1: {result.stage1_time_ms:.3f}ms, Stage2: {result.stage2_time_ms:.3f}ms")
        print(f"    Avg candidates: {result.avg_candidates:.0f}")

    # =========================================================================
    # FAISS TWO-STAGE
    # =========================================================================
    print("\n" + "-" * 70)
    print("FAISS TWO-STAGE")
    print("-" * 70)

    import faiss

    faiss_configs = [
        ("Flat", 50),
        ("Flat", 100),
        ("Flat", 200),
        ("IVF", 50),
        ("IVF", 100),
        ("IVF", 200),
    ]

    for method, k_coarse in faiss_configs:
        print(f"\n  Config: {method}, k_coarse={k_coarse}")

        # Build
        build_start = time.time()
        if method == "Flat":
            faiss_index = faiss.IndexFlatL2(dim)
            faiss_index.add(index_vectors)
        else:  # IVF
            nlist = min(int(np.sqrt(n_index)), 256)
            quantizer = faiss.IndexFlatL2(dim)
            faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            faiss_index.train(index_vectors)
            faiss_index.add(index_vectors)
            faiss_index.nprobe = 10
        build_time = time.time() - build_start
        print(f"    Build time: {build_time:.2f}s")

        # Query
        all_predictions = []
        total_stage1_time = 0
        total_stage2_time = 0

        for query in query_vectors:
            # Stage 1: Coarse
            candidates, s1_time = faiss_coarse_search(query, faiss_index, k=k_coarse)
            total_stage1_time += s1_time

            # Stage 2: Rerank (same as thermal)
            final, s2_time = rerank_exact(query, candidates, index_vectors, k=10)
            total_stage2_time += s2_time

            all_predictions.append(final)

        all_predictions = np.array(all_predictions)

        recall_1 = compute_recall(all_predictions, ground_truth, 1)
        recall_10 = compute_recall(all_predictions, ground_truth, 10)

        result = TwoStageResult(
            method=f"FAISS_{method}_k{k_coarse}",
            recall_at_1=recall_1,
            recall_at_10=recall_10,
            stage1_time_ms=total_stage1_time * 1000 / n_query,
            stage2_time_ms=total_stage2_time * 1000 / n_query,
            total_time_ms=(total_stage1_time + total_stage2_time) * 1000 / n_query,
            avg_candidates=k_coarse,
            n_queries=n_query,
            config={"method": method, "k_coarse": k_coarse}
        )
        results.append(result)

        print(f"    Recall@1: {recall_1*100:.1f}%, Recall@10: {recall_10*100:.1f}%")
        print(f"    Stage1: {result.stage1_time_ms:.3f}ms, Stage2: {result.stage2_time_ms:.3f}ms")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<40} {'R@1':>7} {'R@10':>7} {'S1(ms)':>8} {'S2(ms)':>8} {'Total':>8} {'Cands':>7}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: -x.recall_at_1):
        print(f"{r.method:<40} {r.recall_at_1*100:>6.1f}% {r.recall_at_10*100:>6.1f}% "
              f"{r.stage1_time_ms:>8.3f} {r.stage2_time_ms:>8.3f} {r.total_time_ms:>8.3f} {r.avg_candidates:>7.0f}")

    # Analysis
    print("\n" + "-" * 70)
    print("KEY COMPARISONS")
    print("-" * 70)

    thermal_results = [r for r in results if 'Thermal' in r.method]
    faiss_results = [r for r in results if 'FAISS' in r.method]

    best_thermal = max(thermal_results, key=lambda x: x.recall_at_1)
    best_faiss = max(faiss_results, key=lambda x: x.recall_at_1)
    faiss_flat_100 = next(r for r in results if r.method == 'FAISS_Flat_k100')

    print(f"\nBest Thermal: {best_thermal.method}")
    print(f"  Recall@1: {best_thermal.recall_at_1*100:.1f}%")
    print(f"  Recall@10: {best_thermal.recall_at_10*100:.1f}%")
    print(f"  Stage 1 time: {best_thermal.stage1_time_ms:.3f}ms")

    print(f"\nFAISS Flat (k=100) baseline:")
    print(f"  Recall@1: {faiss_flat_100.recall_at_1*100:.1f}%")
    print(f"  Recall@10: {faiss_flat_100.recall_at_10*100:.1f}%")
    print(f"  Stage 1 time: {faiss_flat_100.stage1_time_ms:.3f}ms")

    speedup = faiss_flat_100.stage1_time_ms / best_thermal.stage1_time_ms
    print(f"\nStage 1 Speedup: {speedup:.1f}x (Thermal vs FAISS Flat)")

    # Success criteria
    print("\n" + "-" * 70)
    print("SUCCESS CRITERIA")
    print("-" * 70)
    print("Target: ≥90% of FAISS recall, ≥5x faster Stage 1")
    print(f"Recall ratio: {best_thermal.recall_at_1 / faiss_flat_100.recall_at_1 * 100:.1f}%")
    print(f"Stage 1 speedup: {speedup:.1f}x")

    meets_recall = best_thermal.recall_at_1 >= 0.9 * faiss_flat_100.recall_at_1
    meets_speed = speedup >= 5.0

    print(f"\n[{'PASS' if meets_recall else 'FAIL'}] Recall ≥ 90% of FAISS")
    print(f"[{'PASS' if meets_speed else 'FAIL'}] Stage 1 ≥ 5x faster")

    # Save
    results_dict = [asdict(r) for r in results]
    with open('two_stage_benchmark_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("\nSaved: two_stage_benchmark_results.json")

    return results


if __name__ == '__main__':
    run_two_stage_benchmark(n_index=100000, n_query=10000, dim=384, n_clusters=100)
