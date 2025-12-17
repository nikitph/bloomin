#!/usr/bin/env python3
"""
Thermal Bloom vs FAISS Benchmark
================================
Compares Thermal Bloom Filter against state-of-the-art FAISS on real embeddings.

Success criterion: ≥95% recall at ≥10x speed = Nature/NeurIPS paper
"""

import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# THERMAL BLOOM V2 (Python implementation for fair comparison)
# ============================================================================

class ThermalBloomV2:
    """Thermal Bloom Filter with multi-item storage and candidate ranking"""

    def __init__(self, grid_size: int = 256, sigma: float = 1.0,
                 range_min: float = -1.0, range_max: float = 1.0):
        self.grid_size = grid_size
        self.sigma = sigma
        self.range_min = range_min
        self.range_max = range_max
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.items = {}  # (x, y) -> list of item indices
        self.points = None  # Original 2D points for distance computation
        self.original_vectors = None  # Original high-D vectors

    def _hash(self, point: np.ndarray) -> Tuple[int, int]:
        """Map 2D point to grid cell"""
        scale = (self.grid_size - 1) / (self.range_max - self.range_min)
        x = int(np.clip((point[0] - self.range_min) * scale, 0, self.grid_size - 1))
        y = int(np.clip((point[1] - self.range_min) * scale, 0, self.grid_size - 1))
        return x, y

    def build(self, points_2d: np.ndarray, original_vectors: np.ndarray = None):
        """Build the thermal bloom index"""
        self.points = points_2d
        self.original_vectors = original_vectors

        # Insert all points
        for i, point in enumerate(points_2d):
            x, y = self._hash(point)
            self.grid[x, y] = 1.0
            if (x, y) not in self.items:
                self.items[(x, y)] = []
            self.items[(x, y)].append(i)

        # Apply thermal diffusion (Gaussian blur)
        self.grid = gaussian_filter(self.grid, sigma=self.sigma)

    def query(self, query_point: np.ndarray, query_vector: np.ndarray = None,
              max_steps: int = 50, search_radius: int = 3) -> Tuple[Optional[int], int, int]:
        """
        Query with gradient ascent and candidate ranking.
        Returns: (best_match_idx, num_steps, num_candidates)
        """
        x, y = self._hash(query_point)
        visited = [(x, y)]

        # Gradient ascent
        for step in range(max_steps):
            if x <= 0 or x >= self.grid_size - 1 or y <= 0 or y >= self.grid_size - 1:
                break

            # Compute gradient
            dx = (self.grid[x + 1, y] - self.grid[x - 1, y]) / 2.0
            dy = (self.grid[x, y + 1] - self.grid[x, y - 1]) / 2.0

            grad_mag = np.sqrt(dx * dx + dy * dy)
            if grad_mag < 1e-8:
                break

            # Move uphill
            new_x = x + (1 if dx > 0.001 else (-1 if dx < -0.001 else 0))
            new_y = y + (1 if dy > 0.001 else (-1 if dy < -0.001 else 0))

            if new_x == x and new_y == y:
                break

            x, y = new_x, new_y
            visited.append((x, y))

        # Collect candidates from neighborhood
        candidates = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx = np.clip(x + dx, 0, self.grid_size - 1)
                ny = np.clip(y + dy, 0, self.grid_size - 1)
                if (nx, ny) in self.items:
                    candidates.extend(self.items[(nx, ny)])

        # Also collect from visited path
        for vx, vy in visited:
            if (vx, vy) in self.items:
                candidates.extend(self.items[(vx, vy)])

        candidates = list(set(candidates))
        num_candidates = len(candidates)

        if not candidates:
            return None, len(visited), 0

        # Rank by distance (use original vectors if available, else 2D points)
        if query_vector is not None and self.original_vectors is not None:
            distances = [np.sum((query_vector - self.original_vectors[c]) ** 2)
                        for c in candidates]
        else:
            distances = [np.sum((query_point - self.points[c]) ** 2)
                        for c in candidates]

        best_idx = candidates[np.argmin(distances)]
        return best_idx, len(visited), num_candidates

    def batch_query(self, query_points: np.ndarray, query_vectors: np.ndarray = None,
                    max_steps: int = 50, search_radius: int = 3) -> np.ndarray:
        """Batch query for multiple points"""
        results = []
        for i in range(len(query_points)):
            qv = query_vectors[i] if query_vectors is not None else None
            best, _, _ = self.query(query_points[i], qv, max_steps, search_radius)
            results.append(best if best is not None else -1)
        return np.array(results)


# ============================================================================
# HIERARCHICAL THERMAL BLOOM (Multi-resolution for high-D)
# ============================================================================

class HierarchicalThermalBloom:
    """
    Hierarchical Thermal Bloom for high-dimensional data.
    Uses multiple 2D projections to capture different aspects of the data.
    """

    def __init__(self, n_projections: int = 4, grid_size: int = 256,
                 sigma: float = 1.0, search_radius: int = 3):
        self.n_projections = n_projections
        self.grid_size = grid_size
        self.sigma = sigma
        self.search_radius = search_radius
        self.blooms = []
        self.projections = []
        self.original_vectors = None

    def build(self, vectors: np.ndarray):
        """Build hierarchical index with multiple random projections"""
        self.original_vectors = vectors
        n_samples, dim = vectors.shape

        # Create multiple random 2D projections
        np.random.seed(42)
        for i in range(self.n_projections):
            # Random projection matrix
            proj_matrix = np.random.randn(dim, 2).astype(np.float32)
            proj_matrix /= np.linalg.norm(proj_matrix, axis=0)

            # Project data
            projected = vectors @ proj_matrix

            # Normalize to [-1, 1]
            proj_min, proj_max = projected.min(axis=0), projected.max(axis=0)
            projected = 2 * (projected - proj_min) / (proj_max - proj_min + 1e-8) - 1

            # Build thermal bloom for this projection
            bloom = ThermalBloomV2(
                grid_size=self.grid_size,
                sigma=self.sigma,
                range_min=-1.1,
                range_max=1.1
            )
            bloom.build(projected, vectors)

            self.blooms.append(bloom)
            self.projections.append((proj_matrix, proj_min, proj_max))

    def query(self, query_vector: np.ndarray, k: int = 1) -> np.ndarray:
        """Query using all projections and aggregate candidates"""
        all_candidates = set()

        for bloom, (proj_matrix, proj_min, proj_max) in zip(self.blooms, self.projections):
            # Project query
            projected = query_vector @ proj_matrix
            projected = 2 * (projected - proj_min) / (proj_max - proj_min + 1e-8) - 1

            # Get candidates from this bloom
            _, _, _ = bloom.query(projected, query_vector,
                                  max_steps=50, search_radius=self.search_radius)

            # Collect all candidates from this bloom's items
            x, y = bloom._hash(projected)
            for dx in range(-self.search_radius * 2, self.search_radius * 2 + 1):
                for dy in range(-self.search_radius * 2, self.search_radius * 2 + 1):
                    nx = np.clip(x + dx, 0, bloom.grid_size - 1)
                    ny = np.clip(y + dy, 0, bloom.grid_size - 1)
                    if (nx, ny) in bloom.items:
                        all_candidates.update(bloom.items[(nx, ny)])

        if not all_candidates:
            return np.array([-1] * k)

        # Rank all candidates by actual distance in original space
        candidates = list(all_candidates)
        distances = [np.sum((query_vector - self.original_vectors[c]) ** 2)
                    for c in candidates]
        sorted_idx = np.argsort(distances)[:k]

        result = [candidates[i] for i in sorted_idx]
        while len(result) < k:
            result.append(-1)
        return np.array(result)

    def batch_query(self, query_vectors: np.ndarray, k: int = 1) -> np.ndarray:
        """Batch query for multiple vectors"""
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
    index_size_mb: float
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
        pred_set = set(p for p in pred[:k] if p >= 0)
        if pred_set & gt_set:
            hits += 1
    return hits / len(predictions)

def compute_ground_truth(index_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute exact k-NN ground truth using brute force"""
    print(f"  Computing ground truth k-NN (k={k})...")
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    nn.fit(index_vectors)
    _, indices = nn.kneighbors(query_vectors)
    return indices


# ============================================================================
# FAISS BENCHMARKS
# ============================================================================

def benchmark_faiss_flat(index_vectors: np.ndarray, query_vectors: np.ndarray,
                         ground_truth: np.ndarray) -> BenchmarkResult:
    """Benchmark FAISS Flat (exact search) - baseline"""
    import faiss

    d = index_vectors.shape[1]

    # Build index
    start = time.time()
    index = faiss.IndexFlatL2(d)
    index.add(index_vectors.astype('float32'))
    build_time = time.time() - start

    # Query
    start = time.time()
    D, I = index.search(query_vectors.astype('float32'), k=10)
    query_time = time.time() - start

    recall_1 = compute_recall(I[:, 0], ground_truth, 1)
    recall_10 = compute_recall(I, ground_truth, 10)
    qps = len(query_vectors) / query_time

    return BenchmarkResult(
        method="FAISS_Flat",
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        queries_per_second=qps,
        build_time_sec=build_time,
        index_size_mb=index_vectors.nbytes / 1e6,
        n_index=len(index_vectors),
        n_query=len(query_vectors),
        config={"type": "exact"}
    )

def benchmark_faiss_ivf(index_vectors: np.ndarray, query_vectors: np.ndarray,
                        ground_truth: np.ndarray, nlist: int = 100,
                        nprobe: int = 10) -> BenchmarkResult:
    """Benchmark FAISS IVF (inverted file index)"""
    import faiss

    d = index_vectors.shape[1]

    # Build index
    start = time.time()
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(index_vectors.astype('float32'))
    index.add(index_vectors.astype('float32'))
    index.nprobe = nprobe
    build_time = time.time() - start

    # Query
    start = time.time()
    D, I = index.search(query_vectors.astype('float32'), k=10)
    query_time = time.time() - start

    recall_1 = compute_recall(I[:, 0], ground_truth, 1)
    recall_10 = compute_recall(I, ground_truth, 10)
    qps = len(query_vectors) / query_time

    return BenchmarkResult(
        method=f"FAISS_IVF_nlist{nlist}_nprobe{nprobe}",
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        queries_per_second=qps,
        build_time_sec=build_time,
        index_size_mb=index_vectors.nbytes / 1e6,
        n_index=len(index_vectors),
        n_query=len(query_vectors),
        config={"nlist": nlist, "nprobe": nprobe}
    )

def benchmark_faiss_hnsw(index_vectors: np.ndarray, query_vectors: np.ndarray,
                         ground_truth: np.ndarray, M: int = 32,
                         ef_search: int = 64) -> BenchmarkResult:
    """Benchmark FAISS HNSW (Hierarchical Navigable Small World)"""
    import faiss

    d = index_vectors.shape[1]

    # Build index
    start = time.time()
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efSearch = ef_search
    index.add(index_vectors.astype('float32'))
    build_time = time.time() - start

    # Query
    start = time.time()
    D, I = index.search(query_vectors.astype('float32'), k=10)
    query_time = time.time() - start

    recall_1 = compute_recall(I[:, 0], ground_truth, 1)
    recall_10 = compute_recall(I, ground_truth, 10)
    qps = len(query_vectors) / query_time

    return BenchmarkResult(
        method=f"FAISS_HNSW_M{M}_ef{ef_search}",
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        queries_per_second=qps,
        build_time_sec=build_time,
        index_size_mb=index_vectors.nbytes / 1e6 * 1.5,  # HNSW has ~50% overhead
        n_index=len(index_vectors),
        n_query=len(query_vectors),
        config={"M": M, "ef_search": ef_search}
    )


# ============================================================================
# THERMAL BLOOM BENCHMARKS
# ============================================================================

def benchmark_thermal_2d_pca(index_vectors: np.ndarray, query_vectors: np.ndarray,
                              ground_truth: np.ndarray, grid_size: int = 256,
                              sigma: float = 1.0, search_radius: int = 3) -> BenchmarkResult:
    """Benchmark Thermal Bloom with PCA reduction to 2D"""
    # PCA reduction
    start = time.time()
    pca = PCA(n_components=2)
    index_2d = pca.fit_transform(index_vectors)
    query_2d = pca.transform(query_vectors)

    # Normalize
    all_2d = np.vstack([index_2d, query_2d])
    min_val, max_val = all_2d.min(), all_2d.max()
    index_2d = 2 * (index_2d - min_val) / (max_val - min_val + 1e-8) - 1
    query_2d = 2 * (query_2d - min_val) / (max_val - min_val + 1e-8) - 1

    # Build thermal bloom
    bloom = ThermalBloomV2(grid_size=grid_size, sigma=sigma, range_min=-1.1, range_max=1.1)
    bloom.build(index_2d, index_vectors)
    build_time = time.time() - start

    # Query
    start = time.time()
    results = bloom.batch_query(query_2d, query_vectors, max_steps=50, search_radius=search_radius)
    query_time = time.time() - start

    recall_1 = compute_recall(results, ground_truth, 1)
    recall_10 = 0  # Single result
    qps = len(query_vectors) / query_time

    return BenchmarkResult(
        method=f"Thermal_PCA2D_g{grid_size}_s{sigma}_r{search_radius}",
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        queries_per_second=qps,
        build_time_sec=build_time,
        index_size_mb=(grid_size * grid_size * 4 + index_vectors.nbytes) / 1e6,
        n_index=len(index_vectors),
        n_query=len(query_vectors),
        config={"grid_size": grid_size, "sigma": sigma, "search_radius": search_radius, "projection": "PCA"}
    )

def benchmark_thermal_hierarchical(index_vectors: np.ndarray, query_vectors: np.ndarray,
                                    ground_truth: np.ndarray, n_projections: int = 4,
                                    grid_size: int = 256, sigma: float = 1.0,
                                    search_radius: int = 5) -> BenchmarkResult:
    """Benchmark Hierarchical Thermal Bloom with multiple projections"""
    # Build
    start = time.time()
    htb = HierarchicalThermalBloom(
        n_projections=n_projections,
        grid_size=grid_size,
        sigma=sigma,
        search_radius=search_radius
    )
    htb.build(index_vectors)
    build_time = time.time() - start

    # Query
    start = time.time()
    results = htb.batch_query(query_vectors, k=1)
    query_time = time.time() - start

    recall_1 = compute_recall(results[:, 0], ground_truth, 1)
    recall_10 = 0
    qps = len(query_vectors) / query_time

    return BenchmarkResult(
        method=f"Thermal_Hier_p{n_projections}_g{grid_size}_s{sigma}_r{search_radius}",
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        queries_per_second=qps,
        build_time_sec=build_time,
        index_size_mb=(n_projections * grid_size * grid_size * 4 + index_vectors.nbytes) / 1e6,
        n_index=len(index_vectors),
        n_query=len(query_vectors),
        config={"n_projections": n_projections, "grid_size": grid_size,
                "sigma": sigma, "search_radius": search_radius}
    )


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def generate_synthetic_embeddings(n_samples: int = 100000, dim: int = 384,
                                   n_clusters: int = 100, seed: int = 42) -> np.ndarray:
    """Generate synthetic embeddings mimicking sentence embeddings"""
    np.random.seed(seed)

    # Generate cluster centers
    centers = np.random.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Generate samples around centers
    samples = []
    for i in range(n_samples):
        center = centers[i % n_clusters]
        noise = np.random.randn(dim) * 0.3
        sample = center + noise
        sample /= np.linalg.norm(sample)
        samples.append(sample)

    return np.array(samples, dtype=np.float32)

def load_real_embeddings(max_samples: int = 100000):
    """Load real sentence embeddings"""
    try:
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset

        print("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        print(f"Loading dataset (max {max_samples} samples)...")
        # Use a smaller, faster dataset for initial testing
        dataset = load_dataset('ag_news', split=f'train[:{max_samples}]')
        texts = dataset['text']

        print("Generating embeddings...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)

        return embeddings.astype(np.float32)

    except ImportError as e:
        print(f"Could not load real embeddings: {e}")
        print("Falling back to synthetic embeddings...")
        return None

def run_full_benchmark(n_index: int = 50000, n_query: int = 5000,
                       dim: int = 384, use_real: bool = True):
    """Run full benchmark comparing all methods"""
    print("=" * 70)
    print("THERMAL BLOOM vs FAISS - FULL BENCHMARK")
    print("=" * 70)

    # Load or generate embeddings
    if use_real:
        embeddings = load_real_embeddings(n_index + n_query)
        if embeddings is None:
            embeddings = generate_synthetic_embeddings(n_index + n_query, dim)
            print(f"Using synthetic embeddings: {embeddings.shape}")
        else:
            print(f"Using real embeddings: {embeddings.shape}")
    else:
        embeddings = generate_synthetic_embeddings(n_index + n_query, dim)
        print(f"Using synthetic embeddings: {embeddings.shape}")

    # Split into index and query
    index_vectors = embeddings[:n_index]
    query_vectors = embeddings[n_index:n_index + n_query]

    print(f"\nIndex: {len(index_vectors)} vectors, Query: {len(query_vectors)} vectors")
    print(f"Dimension: {index_vectors.shape[1]}")

    # Compute ground truth
    ground_truth = compute_ground_truth(index_vectors, query_vectors, k=10)

    results = []

    # FAISS benchmarks
    print("\n" + "-" * 70)
    print("FAISS BENCHMARKS")
    print("-" * 70)

    try:
        import faiss
        print("\nRunning FAISS Flat (exact)...")
        results.append(benchmark_faiss_flat(index_vectors, query_vectors, ground_truth))
        print(f"  Recall@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

        print("\nRunning FAISS IVF (nlist=100, nprobe=10)...")
        results.append(benchmark_faiss_ivf(index_vectors, query_vectors, ground_truth, 100, 10))
        print(f"  Recall@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

        print("\nRunning FAISS IVF (nlist=100, nprobe=50)...")
        results.append(benchmark_faiss_ivf(index_vectors, query_vectors, ground_truth, 100, 50))
        print(f"  Recall@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

        print("\nRunning FAISS HNSW (M=32, ef=64)...")
        results.append(benchmark_faiss_hnsw(index_vectors, query_vectors, ground_truth, 32, 64))
        print(f"  Recall@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

        print("\nRunning FAISS HNSW (M=32, ef=128)...")
        results.append(benchmark_faiss_hnsw(index_vectors, query_vectors, ground_truth, 32, 128))
        print(f"  Recall@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

    except ImportError:
        print("FAISS not installed. Skipping FAISS benchmarks.")
        print("Install with: pip install faiss-cpu")

    # Thermal Bloom benchmarks
    print("\n" + "-" * 70)
    print("THERMAL BLOOM BENCHMARKS")
    print("-" * 70)

    # PCA-based 2D projection
    configs = [
        (256, 0.5, 3),
        (256, 1.0, 5),
        (512, 0.5, 3),
        (512, 1.0, 5),
    ]

    for grid_size, sigma, search_radius in configs:
        print(f"\nRunning Thermal PCA-2D (grid={grid_size}, σ={sigma}, r={search_radius})...")
        results.append(benchmark_thermal_2d_pca(
            index_vectors, query_vectors, ground_truth,
            grid_size=grid_size, sigma=sigma, search_radius=search_radius
        ))
        print(f"  Recall@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

    # Hierarchical thermal bloom
    hier_configs = [
        (4, 256, 1.0, 5),
        (8, 256, 1.0, 5),
        (4, 512, 0.5, 5),
        (8, 512, 0.5, 5),
    ]

    for n_proj, grid_size, sigma, search_radius in hier_configs:
        print(f"\nRunning Thermal Hierarchical (proj={n_proj}, grid={grid_size}, σ={sigma}, r={search_radius})...")
        results.append(benchmark_thermal_hierarchical(
            index_vectors, query_vectors, ground_truth,
            n_projections=n_proj, grid_size=grid_size, sigma=sigma, search_radius=search_radius
        ))
        print(f"  Recall@1: {results[-1].recall_at_1*100:.1f}%, QPS: {results[-1].queries_per_second:,.0f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<50} {'R@1':>8} {'R@10':>8} {'QPS':>12} {'Build(s)':>10}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: -x.recall_at_1):
        print(f"{r.method:<50} {r.recall_at_1*100:>7.1f}% {r.recall_at_10*100:>7.1f}% {r.queries_per_second:>12,.0f} {r.build_time_sec:>10.2f}")

    # Find best thermal result
    thermal_results = [r for r in results if 'Thermal' in r.method]
    faiss_results = [r for r in results if 'FAISS' in r.method]

    if thermal_results and faiss_results:
        best_thermal = max(thermal_results, key=lambda x: x.recall_at_1)
        best_faiss = max(faiss_results, key=lambda x: x.recall_at_1)
        faiss_flat = next((r for r in results if r.method == 'FAISS_Flat'), None)

        print("\n" + "-" * 70)
        print("KEY COMPARISONS")
        print("-" * 70)
        print(f"\nBest Thermal: {best_thermal.method}")
        print(f"  Recall@1: {best_thermal.recall_at_1*100:.1f}%")
        print(f"  QPS: {best_thermal.queries_per_second:,.0f}")

        print(f"\nBest FAISS (approx): {best_faiss.method}")
        print(f"  Recall@1: {best_faiss.recall_at_1*100:.1f}%")
        print(f"  QPS: {best_faiss.queries_per_second:,.0f}")

        if faiss_flat:
            speedup = best_thermal.queries_per_second / faiss_flat.queries_per_second
            print(f"\nSpeedup vs FAISS Flat: {speedup:.1f}x")
            print(f"Recall gap vs exact: {(faiss_flat.recall_at_1 - best_thermal.recall_at_1)*100:.1f}%")

        # Success criteria
        print("\n" + "-" * 70)
        print("SUCCESS CRITERIA (Nature/NeurIPS level)")
        print("-" * 70)
        target_recall = 0.95
        target_speedup = 10.0

        if faiss_flat:
            actual_speedup = best_thermal.queries_per_second / faiss_flat.queries_per_second
            meets_recall = best_thermal.recall_at_1 >= target_recall
            meets_speed = actual_speedup >= target_speedup

            print(f"  Target: ≥95% recall at ≥10x speedup vs exact search")
            print(f"  Achieved: {best_thermal.recall_at_1*100:.1f}% recall at {actual_speedup:.1f}x speedup")
            print(f"  [{'PASS' if meets_recall else 'FAIL'}] Recall ≥ 95%: {best_thermal.recall_at_1*100:.1f}%")
            print(f"  [{'PASS' if meets_speed else 'FAIL'}] Speedup ≥ 10x: {actual_speedup:.1f}x")

    # Save results
    results_dict = [
        {
            'method': r.method,
            'recall_at_1': r.recall_at_1,
            'recall_at_10': r.recall_at_10,
            'queries_per_second': r.queries_per_second,
            'build_time_sec': r.build_time_sec,
            'index_size_mb': r.index_size_mb,
            'n_index': r.n_index,
            'n_query': r.n_query,
            'config': r.config
        }
        for r in results
    ]

    with open('faiss_benchmark_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("\nSaved: faiss_benchmark_results.json")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-index', type=int, default=50000)
    parser.add_argument('--n-query', type=int, default=5000)
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data only')
    args = parser.parse_args()

    run_full_benchmark(
        n_index=args.n_index,
        n_query=args.n_query,
        use_real=not args.synthetic
    )
