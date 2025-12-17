# Thermal Bloom Filters: Differentiable Indexing via Controlled Information Diffusion

## Abstract

We introduce *thermal bloom filters*, a novel data structure that combines discrete hashing with continuous thermodynamic diffusion to enable gradient-guided approximate retrieval. Unlike traditional bloom filters which provide binary set membership, our approach creates a continuous potential field that guides queries toward stored items via gradient ascent. The key insight is that controlled "information leakage" through Gaussian diffusion transforms a discrete hash table into a differentiable optimization landscape, where queries can follow local gradients to find approximate nearest neighbors without tree traversal or graph navigation.

On 2D synthetic benchmarks with 8,000 indexed points, we achieve **99.6% recall@1** compared to 17.6% for discrete bloom filters—a **5.7× improvement**—with gradient descent converging in an average of 12 steps. We derive a universal scaling law relating diffusion width σ to grid resolution, showing that the optimal ratio σ/Δx remains constant across grid sizes, a hallmark of fundamental physical principles.

We demonstrate that thermal bloom filters occupy a distinct region of the algorithm design space: they dominate tree-based methods for inherently low-dimensional data (2D–3D spatial queries, feature map indexing, semantic caching) while complementing rather than replacing high-dimensional approximate nearest neighbor methods like HNSW. Our analysis establishes thermal diffusion as a general technique for making discrete data structures differentiable.

## Key Contributions

1. **Novel Primitive**: First demonstration that hash tables can be made differentiable via thermodynamic diffusion
2. **Near-Perfect Recall**: 99.6% recall@1 on 2D benchmarks with O(k) query complexity
3. **Universal Scaling Law**: σ_optimal/grid_spacing ≈ constant, regardless of dataset size
4. **Clear Design Space**: Precise characterization of when thermal methods dominate tree-based methods

## Significance

This work introduces a fundamentally new approach to approximate retrieval. Rather than building explicit graph structures (HNSW) or hierarchical partitions (IVF), we allow the data itself to diffuse into a continuous field that can be navigated via gradient descent. The approach is particularly relevant for:

- **Spatial Computing**: GPS, mapping, location-based services
- **Scientific Simulation**: Molecular docking, point clouds, particle systems
- **Neural Network Caching**: Semantic similarity in feature space
- **Real-time Systems**: O(1) insertion with amortized diffusion

The core principle—that discrete structures become differentiable through controlled information leakage—may have applications beyond indexing, including learned data structures, continuous relaxations of combinatorial optimization, and neural-symbolic integration.

---

**Keywords**: approximate nearest neighbor, bloom filter, thermodynamic diffusion, gradient descent, spatial indexing

**Code**: Available at [repository URL]

**Figures**:
- Figure 1: Three-panel visualization showing (A) discrete bloom with no gradient, (B) thermal field with gradient vectors, (C) query following gradient to result
- Figure 2: Parameter sensitivity heatmap
- Figure 3: Recall vs diffusion parameter across grid sizes
