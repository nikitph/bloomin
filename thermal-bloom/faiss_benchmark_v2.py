#!/usr/bin/env python3
"""
Thermal Bloom vs FAISS Benchmark V2
===================================
Advanced approaches for high-dimensional embeddings:
1. Multi-scale hierarchical thermal bloom with many projections
2. LSH-based thermal bloom (thermal diffusion in hash space)
3. Hybrid FAISS + Thermal refinement
4. Sparse high-dimensional thermal fields
"""

import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ADVANCED THERMAL BLOOM: SPARSE HIGH-D VERSION
# ============================================================================

class SparseThermalBloom:
    """
    Thermal bloom that operates in higher dimensions using sparse representation.
    Instead of a dense grid, we use a hash-based sparse representation.
    """

    def __init__(self, n_bits: int = 12, sigma_cells: float = 2.0, n_hashes: int = 4):
        self.n_bits = n_bits  # Resolution per dimension
        self.n_cells = 2 ** n_bits
        self.sigma_cells = sigma_cells
        self.n_hashes = n_hashes
        self.hash_tables: List[Dict[int, List[int]]] = [dict() for _ in range(n_hashes)]
        self.thermal_fields: List[Dict[int, float]] = [dict() for _ in range(n_hashes)]
        self.vectors = None
        self.hash_vectors = None  # Random vectors for hashing

    def _compute_hash(self, vector: np.ndarray, hash_idx: int) -> int:
        """Compute locality-sensitive hash for a vector"""
        # Use random hyperplane LSH
        projection = np.dot(vector, self.hash_vectors[hash_idx])
        # Quantize to n_bits
        bits = (projection > 0).astype(int)
        return int(sum(b << i for i, b in enumerate(bits)))

    def build(self, vectors: np.ndarray):
        """Build the sparse thermal bloom index"""
        self.vectors = vectors
        n_samples, dim = vectors.shape

        # Generate random hash vectors (LSH hyperplanes)
        np.random.seed(42)
        self.hash_vectors = [
            np.random.randn(dim, self.n_bits).astype(np.float32)
            for _ in range(self.n_hashes)
        ]

        # Insert all vectors into hash tables
        for i, vec in enumerate(vectors):
            for h in range(self.n_hashes):
                hash_val = self._compute_hash(vec, h)
                if hash_val not in self.hash_tables[h]:
                    self.hash_tables[h][hash_val] = []
                self.hash_tables[h][hash_val].append(i)

        # Apply thermal diffusion in hash space (spread to nearby buckets)
        for h in range(self.n_hashes):
            # For each occupied bucket, spread heat to Hamming neighbors
            for hash_val, items in self.hash_tables[h].items():
                heat = len(items)  # Heat proportional to number of items

                # Spread to Hamming-1 neighbors
                for bit in range(self.n_bits):
                    neighbor = hash_val ^ (1 << bit)
                    if neighbor not in self.thermal_fields[h]:
                        self.thermal_fields[h][neighbor] = 0
                    self.thermal_fields[h][neighbor] += heat * 0.5

                # Self heat
                if hash_val not in self.thermal_fields[h]:
                    self.thermal_fields[h][hash_val] = 0
                self.thermal_fields[h][hash_val] += heat

    def query(self, query_vector: np.ndarray, max_candidates: int = 100) -> Tuple[Optional[int], int]:
        """Query using thermal gradient following in hash space"""
        candidates = set()

        for h in range(self.n_hashes):
            hash_val = self._compute_hash(query_vector, h)

            # Get candidates from this bucket
            if hash_val in self.hash_tables[h]:
                candidates.update(self.hash_tables[h][hash_val])

            # Follow thermal gradient to neighbors with more heat
            visited = {hash_val}
            current = hash_val

            for step in range(10):  # Limited gradient steps
                # Find hottest Hamming neighbor
                best_neighbor = None
                best_heat = self.thermal_fields[h].get(current, 0)

                for bit in range(self.n_bits):
                    neighbor = current ^ (1 << bit)
                    if neighbor not in visited:
                        heat = self.thermal_fields[h].get(neighbor, 0)
                        if heat > best_heat:
                            best_heat = heat
                            best_neighbor = neighbor

                if best_neighbor is None:
                    break

                current = best_neighbor
                visited.add(current)

                # Collect candidates from this bucket
                if current in self.hash_tables[h]:
                    candidates.update(self.hash_tables[h][current])

                if len(candidates) >= max_candidates:
                    break

        if not candidates:
            return None, 0

        # Rank by actual distance
        candidates = list(candidates)[:max_candidates]
        distances = [np.sum((query_vector - self.vectors[c]) ** 2) for c in candidates]
        best = candidates[np.argmin(distances)]

        return best, len(candidates)

    def batch_query(self, query_vectors: np.ndarray, max_candidates: int = 100) -> np.ndarray:
        """Batch query"""
        results = []
        for qv in query_vectors:
            best, _ = self.query(qv, max_candidates)
            results.append(best if best is not None else -1)
        return np.array(results)


# ============================================================================
# MULTI-SCALE THERMAL BLOOM WITH LEARNED PROJECTIONS
# ============================================================================

class MultiScaleThermalBloom:
    """
    Multi-scale thermal bloom with aggressive projection.
    Uses many random projections and aggregates results.
    """

    def __init__(self, n_projections: int = 32, grid_size: int = 128,
                 sigma: float = 2.0, search_radius: int = 10):
        self.n_projections = n_projections
        self.grid_size = grid_size
        self.sigma = sigma
        self.search_radius = search_radius
        self.grids = []
        self.items_maps = []
        self.projectors = []
        self.vectors = None

    def build(self, vectors: np.ndarray):
        """Build multi-scale index"""
        self.vectors = vectors
        n_samples, dim = vectors.shape

        # Create multiple random 2D projections
        np.random.seed(42)

        for p in range(self.n_projections):
            # Random projection to 2D
            proj = GaussianRandomProjection(n_components=2, random_state=p)
            projected = proj.fit_transform(vectors)

            # Normalize to grid range
            pmin, pmax = projected.min(), projected.max()
            projected = (projected - pmin) / (pmax - pmin + 1e-8)
            projected = projected * (self.grid_size - 1)

            # Build grid and items map
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            items_map = {}

            for i, pt in enumerate(projected):
                x, y = int(pt[0]), int(pt[1])
                x = np.clip(x, 0, self.grid_size - 1)
                y = np.clip(y, 0, self.grid_size - 1)
                grid[x, y] = 1.0
                if (x, y) not in items_map:
                    items_map[(x, y)] = []
                items_map[(x, y)].append(i)

            # Apply thermal diffusion
            grid = gaussian_filter(grid, sigma=self.sigma)

            self.grids.append(grid)
            self.items_maps.append(items_map)
            self.projectors.append((proj, pmin, pmax))

    def query(self, query_vector: np.ndarray, max_candidates: int = 500) -> Tuple[Optional[int], int]:
        """Query using all projections"""
        all_candidates = set()

        for p in range(self.n_projections):
            proj, pmin, pmax = self.projectors[p]
            grid = self.grids[p]
            items_map = self.items_maps[p]

            # Project query
            projected = proj.transform(query_vector.reshape(1, -1))[0]
            projected = (projected - pmin) / (pmax - pmin + 1e-8)
            projected = projected * (self.grid_size - 1)

            x, y = int(projected[0]), int(projected[1])
            x = np.clip(x, 0, self.grid_size - 1)
            y = np.clip(y, 0, self.grid_size - 1)

            # Gradient ascent
            for step in range(20):
                if x <= 0 or x >= self.grid_size - 1 or y <= 0 or y >= self.grid_size - 1:
                    break

                dx = (grid[x + 1, y] - grid[x - 1, y]) / 2
                dy = (grid[x, y + 1] - grid[x, y - 1]) / 2

                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    break

                x += 1 if dx > 0.001 else (-1 if dx < -0.001 else 0)
                y += 1 if dy > 0.001 else (-1 if dy < -0.001 else 0)

            # Collect candidates from neighborhood
            for dx in range(-self.search_radius, self.search_radius + 1):
                for dy in range(-self.search_radius, self.search_radius + 1):
                    nx = np.clip(x + dx, 0, self.grid_size - 1)
                    ny = np.clip(y + dy, 0, self.grid_size - 1)
                    if (nx, ny) in items_map:
                        all_candidates.update(items_map[(nx, ny)])

            if len(all_candidates) >= max_candidates:
                break

        if not all_candidates:
            return None, 0

        # Rank by distance
        candidates = list(all_candidates)[:max_candidates]
        distances = [np.sum((query_vector - self.vectors[c]) ** 2) for c in candidates]
        best = candidates[np.argmin(distances)]

        return best, len(candidates)

    def batch_query(self, query_vectors: np.ndarray, max_candidates: int = 500) -> np.ndarray:
        """Batch query"""
        results = []
        for qv in query_vectors:
            best, _ = self.query(qv, max_candidates)
            results.append(best if best is not None else -1)
        return np.array(results)


# ============================================================================
# HYBRID FAISS + THERMAL REFINEMENT
# ============================================================================

class HybridFAISSThermal:
    """
    Hybrid approach: FAISS for coarse search, Thermal for refinement.
    """

    def __init__(self, n_coarse: int = 100, grid_size: int = 256, sigma: float = 1.0):
        self.n_coarse = n_coarse  # Number of coarse candidates from FAISS
        self.grid_size = grid_size
        self.sigma = sigma
        self.faiss_index = None
        self.thermal_grids = {}  # Cluster-specific thermal grids
        self.cluster_items = {}
        self.vectors = None

    def build(self, vectors: np.ndarray):
        """Build hybrid index"""
        import faiss

        self.vectors = vectors
        n_samples, dim = vectors.shape

        # Build FAISS IVF for coarse search
        nlist = min(int(np.sqrt(n_samples)), 256)
        quantizer = faiss.IndexFlatL2(dim)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        self.faiss_index.train(vectors.astype('float32'))
        self.faiss_index.add(vectors.astype('float32'))
        self.faiss_index.nprobe = 10

        # Get cluster assignments
        _, cluster_ids = quantizer.search(vectors.astype('float32'), 1)

        # Build thermal grid for each cluster using PCA
        for cluster_id in range(nlist):
            cluster_mask = (cluster_ids[:, 0] == cluster_id)
            if not np.any(cluster_mask):
                continue

            cluster_indices = np.where(cluster_mask)[0]
            cluster_vectors = vectors[cluster_mask]

            if len(cluster_vectors) < 3:
                continue

            # PCA to 2D within cluster
            pca = PCA(n_components=min(2, len(cluster_vectors) - 1))
            projected = pca.fit_transform(cluster_vectors)

            if projected.shape[1] < 2:
                continue

            # Normalize
            pmin, pmax = projected.min(), projected.max()
            if pmax - pmin < 1e-8:
                continue
            projected = (projected - pmin) / (pmax - pmin)
            projected = projected * (self.grid_size - 1)

            # Build grid
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            items_map = {}

            for i, (pt, global_idx) in enumerate(zip(projected, cluster_indices)):
                x, y = int(pt[0]), int(pt[1])
                x = np.clip(x, 0, self.grid_size - 1)
                y = np.clip(y, 0, self.grid_size - 1)
                grid[x, y] = 1.0
                if (x, y) not in items_map:
                    items_map[(x, y)] = []
                items_map[(x, y)].append(global_idx)

            grid = gaussian_filter(grid, sigma=self.sigma)

            self.thermal_grids[cluster_id] = (grid, items_map, pca, pmin, pmax)
            self.cluster_items[cluster_id] = set(cluster_indices)

    def query(self, query_vector: np.ndarray, k: int = 10) -> np.ndarray:
        """Query using hybrid approach"""
        # FAISS coarse search
        D, I = self.faiss_index.search(query_vector.reshape(1, -1).astype('float32'), self.n_coarse)
        coarse_candidates = set(I[0][I[0] >= 0])

        # Identify which clusters the candidates belong to
        cluster_candidates = {}
        for idx in coarse_candidates:
            for cluster_id, cluster_items in self.cluster_items.items():
                if idx in cluster_items:
                    if cluster_id not in cluster_candidates:
                        cluster_candidates[cluster_id] = set()
                    cluster_candidates[cluster_id].add(idx)
                    break

        # Thermal refinement in each cluster
        refined_candidates = set()

        for cluster_id, candidates in cluster_candidates.items():
            if cluster_id not in self.thermal_grids:
                refined_candidates.update(candidates)
                continue

            grid, items_map, pca, pmin, pmax = self.thermal_grids[cluster_id]

            # Project query to cluster's 2D space
            projected = pca.transform(query_vector.reshape(1, -1))[0]
            projected = (projected - pmin) / (pmax - pmin + 1e-8)
            projected = projected * (self.grid_size - 1)

            x, y = int(projected[0]), int(projected[1])
            x = np.clip(x, 0, self.grid_size - 1)
            y = np.clip(y, 0, self.grid_size - 1)

            # Gradient ascent
            for step in range(10):
                if x <= 0 or x >= self.grid_size - 1 or y <= 0 or y >= self.grid_size - 1:
                    break

                dx = (grid[x + 1, y] - grid[x - 1, y]) / 2
                dy = (grid[x, y + 1] - grid[x, y - 1]) / 2

                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    break

                x += 1 if dx > 0.001 else (-1 if dx < -0.001 else 0)
                y += 1 if dy > 0.001 else (-1 if dy < -0.001 else 0)

            # Collect from neighborhood
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx = np.clip(x + dx, 0, self.grid_size - 1)
                    ny = np.clip(y + dy, 0, self.grid_size - 1)
                    if (nx, ny) in items_map:
                        refined_candidates.update(items_map[(nx, ny)])

        # Also include coarse candidates
        refined_candidates.update(coarse_candidates)

        if not refined_candidates:
            return np.array([-1] * k)

        # Rank by distance
        candidates = list(refined_candidates)
        distances = [np.sum((query_vector - self.vectors[c]) ** 2) for c in candidates]
        sorted_idx = np.argsort(distances)[:k]

        result = [candidates[i] for i in sorted_idx]
        while len(result) < k:
            result.append(-1)

        return np.array(result)

    def batch_query(self, query_vectors: np.ndarray, k: int = 1) -> np.ndarray:
        """Batch query"""
        results = []
        for qv in query_vectors:
            results.append(self.query(qv, k))
        return np.array(results)


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

@dataclass
class BenchmarkResult:
    method: str
    recall_at_1: float
    recall_at_10: float
    queries_per_second: float
    build_time_sec: float
    n_index: int
    n_query: int
    config: dict


def compute_recall(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k"""
    hits = 0
    for pred, gt in zip(predictions, ground_truth):
        if isinstance(pred, (int, np.integer)):
            pred = [pred]
        gt_set = set(gt[:k].tolist()) if hasattr(gt, '__iter__') else {gt}
        pred_set = set(p for p in (pred[:k] if hasattr(pred, '__iter__') else [pred]) if p >= 0)
        if pred_set & gt_set:
            hits += 1
    return hits / len(predictions)


def compute_ground_truth(index_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute exact k-NN ground truth"""
    print(f"  Computing ground truth k-NN (k={k})...")
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    nn.fit(index_vectors)
    _, indices = nn.kneighbors(query_vectors)
    return indices


def generate_synthetic_embeddings(n_samples: int = 100000, dim: int = 384,
                                   n_clusters: int = 100, seed: int = 42) -> np.ndarray:
    """Generate synthetic embeddings with cluster structure"""
    np.random.seed(seed)

    # Generate cluster centers (normalized)
    centers = np.random.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Generate samples around centers with varying tightness
    samples = []
    for i in range(n_samples):
        center_idx = i % n_clusters
        center = centers[center_idx]
        noise_scale = 0.2 + np.random.rand() * 0.3  # Variable cluster tightness
        noise = np.random.randn(dim) * noise_scale
        sample = center + noise
        sample /= np.linalg.norm(sample)
        samples.append(sample)

    return np.array(samples, dtype=np.float32)


# ============================================================================
# MAIN BENCHMARK V2
# ============================================================================

def run_advanced_benchmark(n_index: int = 50000, n_query: int = 5000, dim: int = 384):
    """Run advanced benchmark with all methods"""
    print("=" * 70)
    print("THERMAL BLOOM V2 vs FAISS - ADVANCED BENCHMARK")
    print("=" * 70)

    # Generate data
    print("\nGenerating synthetic embeddings...")
    embeddings = generate_synthetic_embeddings(n_index + n_query, dim)
    index_vectors = embeddings[:n_index]
    query_vectors = embeddings[n_index:n_index + n_query]

    print(f"Index: {len(index_vectors)} vectors, Query: {len(query_vectors)} vectors")
    print(f"Dimension: {dim}")

    # Ground truth
    ground_truth = compute_ground_truth(index_vectors, query_vectors, k=10)

    results = []

    # FAISS baselines
    print("\n" + "-" * 70)
    print("FAISS BASELINES")
    print("-" * 70)

    import faiss

    # Flat (exact)
    print("\nFAISS Flat (exact)...")
    start = time.time()
    index_flat = faiss.IndexFlatL2(dim)
    index_flat.add(index_vectors.astype('float32'))
    build_time = time.time() - start

    start = time.time()
    D, I = index_flat.search(query_vectors.astype('float32'), k=10)
    query_time = time.time() - start

    results.append(BenchmarkResult(
        method="FAISS_Flat",
        recall_at_1=compute_recall(I[:, 0], ground_truth, 1),
        recall_at_10=compute_recall(I, ground_truth, 10),
        queries_per_second=len(query_vectors) / query_time,
        build_time_sec=build_time,
        n_index=n_index, n_query=n_query,
        config={"type": "exact"}
    ))
    print(f"  R@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

    # HNSW
    for ef in [64, 128, 256]:
        print(f"\nFAISS HNSW (ef={ef})...")
        start = time.time()
        index_hnsw = faiss.IndexHNSWFlat(dim, 32)
        index_hnsw.hnsw.efSearch = ef
        index_hnsw.add(index_vectors.astype('float32'))
        build_time = time.time() - start

        start = time.time()
        D, I = index_hnsw.search(query_vectors.astype('float32'), k=10)
        query_time = time.time() - start

        results.append(BenchmarkResult(
            method=f"FAISS_HNSW_ef{ef}",
            recall_at_1=compute_recall(I[:, 0], ground_truth, 1),
            recall_at_10=compute_recall(I, ground_truth, 10),
            queries_per_second=len(query_vectors) / query_time,
            build_time_sec=build_time,
            n_index=n_index, n_query=n_query,
            config={"ef_search": ef}
        ))
        print(f"  R@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

    # ADVANCED THERMAL METHODS
    print("\n" + "-" * 70)
    print("ADVANCED THERMAL METHODS")
    print("-" * 70)

    # Sparse Thermal Bloom
    for n_bits in [10, 12]:
        for n_hashes in [4, 8]:
            print(f"\nSparse Thermal (bits={n_bits}, hashes={n_hashes})...")
            start = time.time()
            stb = SparseThermalBloom(n_bits=n_bits, n_hashes=n_hashes)
            stb.build(index_vectors)
            build_time = time.time() - start

            start = time.time()
            preds = stb.batch_query(query_vectors, max_candidates=200)
            query_time = time.time() - start

            results.append(BenchmarkResult(
                method=f"Thermal_Sparse_b{n_bits}_h{n_hashes}",
                recall_at_1=compute_recall(preds, ground_truth, 1),
                recall_at_10=0,
                queries_per_second=len(query_vectors) / query_time,
                build_time_sec=build_time,
                n_index=n_index, n_query=n_query,
                config={"n_bits": n_bits, "n_hashes": n_hashes}
            ))
            print(f"  R@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

    # Multi-scale Thermal
    for n_proj in [16, 32, 64]:
        print(f"\nMulti-Scale Thermal (projections={n_proj})...")
        start = time.time()
        mstb = MultiScaleThermalBloom(n_projections=n_proj, grid_size=128, sigma=2.0, search_radius=10)
        mstb.build(index_vectors)
        build_time = time.time() - start

        start = time.time()
        preds = mstb.batch_query(query_vectors, max_candidates=500)
        query_time = time.time() - start

        results.append(BenchmarkResult(
            method=f"Thermal_MultiScale_p{n_proj}",
            recall_at_1=compute_recall(preds, ground_truth, 1),
            recall_at_10=0,
            queries_per_second=len(query_vectors) / query_time,
            build_time_sec=build_time,
            n_index=n_index, n_query=n_query,
            config={"n_projections": n_proj}
        ))
        print(f"  R@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

    # Hybrid FAISS + Thermal
    for n_coarse in [50, 100, 200]:
        print(f"\nHybrid FAISS+Thermal (coarse={n_coarse})...")
        start = time.time()
        hybrid = HybridFAISSThermal(n_coarse=n_coarse, grid_size=128, sigma=1.0)
        hybrid.build(index_vectors)
        build_time = time.time() - start

        start = time.time()
        preds = hybrid.batch_query(query_vectors, k=1)
        query_time = time.time() - start

        results.append(BenchmarkResult(
            method=f"Hybrid_FAISS_Thermal_c{n_coarse}",
            recall_at_1=compute_recall(preds[:, 0], ground_truth, 1),
            recall_at_10=0,
            queries_per_second=len(query_vectors) / query_time,
            build_time_sec=build_time,
            n_index=n_index, n_query=n_query,
            config={"n_coarse": n_coarse}
        ))
        print(f"  R@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<45} {'R@1':>8} {'R@10':>8} {'QPS':>12} {'Build':>10}")
    print("-" * 85)

    for r in sorted(results, key=lambda x: -x.recall_at_1):
        print(f"{r.method:<45} {r.recall_at_1*100:>7.1f}% {r.recall_at_10*100:>7.1f}% {r.queries_per_second:>12,.0f} {r.build_time_sec:>9.2f}s")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    faiss_flat = next(r for r in results if r.method == 'FAISS_Flat')
    thermal_results = [r for r in results if 'Thermal' in r.method or 'Hybrid' in r.method]

    if thermal_results:
        best_thermal = max(thermal_results, key=lambda x: x.recall_at_1)
        print(f"\nBest Thermal Method: {best_thermal.method}")
        print(f"  Recall@1: {best_thermal.recall_at_1*100:.1f}%")
        print(f"  QPS: {best_thermal.queries_per_second:,.0f}")
        print(f"  Speedup vs Flat: {best_thermal.queries_per_second/faiss_flat.queries_per_second:.1f}x")

        # Check success criteria
        print("\n" + "-" * 70)
        print("SUCCESS CRITERIA")
        print("-" * 70)
        print(f"Target: ≥95% recall at ≥10x speedup")
        speedup = best_thermal.queries_per_second / faiss_flat.queries_per_second
        print(f"Achieved: {best_thermal.recall_at_1*100:.1f}% recall at {speedup:.1f}x speedup")

    # Save
    results_dict = [
        {
            'method': r.method,
            'recall_at_1': r.recall_at_1,
            'recall_at_10': r.recall_at_10,
            'qps': r.queries_per_second,
            'build_time': r.build_time_sec,
            'config': r.config
        }
        for r in results
    ]
    with open('faiss_benchmark_v2_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("\nSaved: faiss_benchmark_v2_results.json")


if __name__ == '__main__':
    run_advanced_benchmark(n_index=50000, n_query=5000, dim=384)
