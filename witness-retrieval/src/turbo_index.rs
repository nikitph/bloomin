//! Turbo Witness-LDPC Index: Maximum Speed Optimizations
//!
//! Implements all three key optimizations:
//! 1. SIMD-friendly Hamming distance (vectorized popcount)
//! 2. Early termination via top-k tracking
//! 3. Cache-optimized batch re-ranking

use ahash::AHasher;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};

/// OPTIMIZATION 1: SIMD-friendly Hamming distance
/// Uses u64 operations that compile to efficient POPCNT instructions
#[inline(always)]
fn hamming_distance_fast(a: &[u64], b: &[u64]) -> u32 {
    // Unroll loop for better pipelining
    let mut total = 0u32;
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    // Process 4 u64s at a time (256 bits)
    for i in 0..chunks {
        let idx = i * 4;
        // XOR and popcount - compiler will use POPCNT instruction
        total += (a[idx] ^ b[idx]).count_ones();
        total += (a[idx + 1] ^ b[idx + 1]).count_ones();
        total += (a[idx + 2] ^ b[idx + 2]).count_ones();
        total += (a[idx + 3] ^ b[idx + 3]).count_ones();
    }

    // Handle remaining
    let base = chunks * 4;
    for i in 0..remainder {
        total += (a[base + i] ^ b[base + i]).count_ones();
    }

    total
}

/// OPTIMIZATION 1b: Hamming SIMILARITY (matching 1-bits, not distance)
#[inline(always)]
fn hamming_similarity_fast(a: &[u64], b: &[u64]) -> u32 {
    let mut total = 0u32;
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let idx = i * 4;
        total += (a[idx] & b[idx]).count_ones();
        total += (a[idx + 1] & b[idx + 1]).count_ones();
        total += (a[idx + 2] & b[idx + 2]).count_ones();
        total += (a[idx + 3] & b[idx + 3]).count_ones();
    }

    let base = chunks * 4;
    for i in 0..remainder {
        total += (a[base + i] & b[base + i]).count_ones();
    }

    total
}

/// OPTIMIZATION 2: Early termination with progressive distance
/// Compute distance chunk by chunk, exit early if can't make top-k
#[inline]
fn hamming_distance_early_exit(a: &[u64], b: &[u64], threshold: u32) -> Option<u32> {
    let mut total = 0u32;
    let chunk_size = 4; // Process 256 bits at a time
    let chunks = a.len() / chunk_size;

    for i in 0..chunks {
        let idx = i * chunk_size;
        total += (a[idx] ^ b[idx]).count_ones();
        total += (a[idx + 1] ^ b[idx + 1]).count_ones();
        total += (a[idx + 2] ^ b[idx + 2]).count_ones();
        total += (a[idx + 3] ^ b[idx + 3]).count_ones();

        // Early exit: if we've already exceeded threshold, give up
        if total > threshold {
            return None;
        }
    }

    // Remainder
    let base = chunks * chunk_size;
    for i in base..a.len() {
        total += (a[i] ^ b[i]).count_ones();
    }

    if total <= threshold {
        Some(total)
    } else {
        None
    }
}

/// Binary code stored as contiguous u64 array
#[derive(Clone)]
pub struct TurboCode {
    pub bits: Vec<u64>,
    pub popcount: u32,
}

impl TurboCode {
    pub fn new(code_length: usize) -> Self {
        let n_chunks = (code_length + 63) / 64;
        Self { bits: vec![0u64; n_chunks], popcount: 0 }
    }

    #[inline]
    pub fn set_bit(&mut self, pos: usize) {
        let chunk = pos / 64;
        let bit = pos % 64;
        if chunk < self.bits.len() {
            let mask = 1u64 << bit;
            if self.bits[chunk] & mask == 0 {
                self.bits[chunk] |= mask;
                self.popcount += 1;
            }
        }
    }

    #[inline]
    pub fn similarity(&self, other: &TurboCode) -> u32 {
        hamming_similarity_fast(&self.bits, &other.bits)
    }

    #[inline]
    pub fn distance(&self, other: &TurboCode) -> u32 {
        hamming_distance_fast(&self.bits, &other.bits)
    }

    pub fn set_bit_indices(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.popcount as usize);
        for (chunk_idx, &chunk) in self.bits.iter().enumerate() {
            let mut c = chunk;
            let base = chunk_idx * 64;
            while c != 0 {
                let tz = c.trailing_zeros() as usize;
                indices.push(base + tz);
                c &= c - 1;
            }
        }
        indices
    }
}

/// Fast hash
#[inline]
fn hash_witness(witness: u32, seed: u64, code_length: usize) -> usize {
    let mut hasher = AHasher::default();
    witness.hash(&mut hasher);
    seed.hash(&mut hasher);
    (hasher.finish() as usize) % code_length
}

/// Extract witnesses
fn extract_witnesses(vector: &[f32], num_witnesses: usize) -> Vec<u32> {
    let dim = vector.len();
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_unstable_by_key(|&i| Reverse(OrderedFloat(vector[i].abs())));

    indices.iter().take(num_witnesses).map(|&idx| {
        let sign_bit = if vector[idx] > 0.0 { 1u32 } else { 0u32 };
        (idx as u32) * 2 + sign_bit
    }).collect()
}

/// SIMD-friendly dot product
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // Unrolled for better vectorization
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = a.len() / 4;
    for i in 0..chunks {
        let idx = i * 4;
        sum0 += a[idx] * b[idx];
        sum1 += a[idx + 1] * b[idx + 1];
        sum2 += a[idx + 2] * b[idx + 2];
        sum3 += a[idx + 3] * b[idx + 3];
    }

    let base = chunks * 4;
    for i in base..a.len() {
        sum0 += a[i] * b[i];
    }

    sum0 + sum1 + sum2 + sum3
}

/// Turbo Witness-LDPC Index
pub struct TurboWitnessIndex {
    pub dim: usize,
    pub code_length: usize,
    pub num_hashes: usize,
    pub num_witnesses: usize,
    hash_seeds: Vec<u64>,
    codes: Vec<TurboCode>,
    // Contiguous code storage for cache efficiency
    codes_flat: Vec<u64>,
    chunks_per_code: usize,
    vectors: Option<Vec<Vec<f32>>>,
    inverted_index: Vec<Vec<u32>>,
    pub n_vectors: usize,
}

impl TurboWitnessIndex {
    pub fn new(dim: usize, code_length: usize, num_hashes: usize, num_witnesses: usize) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let hash_seeds: Vec<u64> = (0..num_hashes).map(|_| rng.gen()).collect();
        let chunks_per_code = (code_length + 63) / 64;

        Self {
            dim,
            code_length,
            num_hashes,
            num_witnesses,
            hash_seeds,
            codes: Vec::new(),
            codes_flat: Vec::new(),
            chunks_per_code,
            vectors: None,
            inverted_index: vec![Vec::new(); code_length],
            n_vectors: 0,
        }
    }

    fn encode(&self, vector: &[f32]) -> TurboCode {
        let witnesses = extract_witnesses(vector, self.num_witnesses);
        let mut code = TurboCode::new(self.code_length);

        for witness in witnesses {
            for (k, &seed) in self.hash_seeds.iter().enumerate() {
                let pos = hash_witness(witness, seed.wrapping_add(k as u64), self.code_length);
                code.set_bit(pos);
            }
        }
        code
    }

    pub fn add(&mut self, vectors: Vec<Vec<f32>>, store_vectors: bool) {
        let n = vectors.len();
        self.n_vectors = n;

        // Pre-normalize
        let normalized: Vec<Vec<f32>> = vectors.par_iter().map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-8 { v.clone() } else { v.iter().map(|x| x / norm).collect() }
        }).collect();

        // Encode
        let codes: Vec<TurboCode> = normalized.par_iter().map(|v| self.encode(v)).collect();

        // OPTIMIZATION 3: Create contiguous flat storage for cache efficiency
        self.codes_flat = Vec::with_capacity(n * self.chunks_per_code);
        for code in &codes {
            self.codes_flat.extend_from_slice(&code.bits);
        }

        // Build inverted index
        self.inverted_index = vec![Vec::new(); self.code_length];
        for (vec_id, code) in codes.iter().enumerate() {
            for bit_pos in code.set_bit_indices() {
                self.inverted_index[bit_pos].push(vec_id as u32);
            }
        }

        self.codes = codes;
        if store_vectors {
            self.vectors = Some(normalized);
        }
    }

    /// Get code slice from flat storage (cache-friendly)
    #[inline]
    fn get_code_flat(&self, vec_id: usize) -> &[u64] {
        let start = vec_id * self.chunks_per_code;
        &self.codes_flat[start..start + self.chunks_per_code]
    }

    /// TURBO SEARCH: All optimizations combined
    pub fn search_turbo(&self, query: &[f32], k: usize, n_candidates: usize) -> Vec<(u32, f32)> {
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm < 1e-8 {
            query.to_vec()
        } else {
            query.iter().map(|x| x / query_norm).collect()
        };

        let query_code = self.encode(&query_normalized);
        let query_bits = query_code.set_bit_indices();

        // Stage 1: Fast candidate generation via inverted index
        let mut scores = vec![0u16; self.n_vectors];
        for &bit_pos in &query_bits {
            for &vec_id in &self.inverted_index[bit_pos] {
                scores[vec_id as usize] += 1;
            }
        }

        // Stage 2: Get top candidates (use partial sort)
        let mut candidates: Vec<(u32, u16)> = scores.iter().enumerate()
            .filter(|(_, &s)| s > 0)
            .map(|(i, &s)| (i as u32, s))
            .collect();

        if candidates.len() > n_candidates {
            candidates.select_nth_unstable_by(n_candidates, |a, b| b.1.cmp(&a.1));
            candidates.truncate(n_candidates);
        }

        // OPTIMIZATION 3: Sort by ID for cache-friendly access
        candidates.sort_unstable_by_key(|(id, _)| *id);

        // Stage 3: Re-rank with exact Hamming or cosine
        if let Some(ref vectors) = self.vectors {
            // OPTIMIZATION 2: Early termination with min-heap
            let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, u32)> = BinaryHeap::with_capacity(k + 1);
            let mut k_th_best = f32::NEG_INFINITY;

            for &(vec_id, _) in &candidates {
                let vec = &vectors[vec_id as usize];

                // Compute similarity
                let sim = dot_product(&query_normalized, vec);

                if sim > k_th_best || heap.len() < k {
                    heap.push((Reverse(OrderedFloat(sim)), vec_id));
                    if heap.len() > k {
                        heap.pop();
                    }
                    if heap.len() == k {
                        k_th_best = heap.peek().unwrap().0.0.into_inner();
                    }
                }
            }

            // Extract results
            let mut results: Vec<(u32, f32)> = heap.into_iter()
                .map(|(Reverse(OrderedFloat(sim)), id)| (id, sim))
                .collect();
            results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            results
        } else {
            // Hamming-only with early termination
            let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
            let mut k_th_worst_distance = u32::MAX;

            for &(vec_id, _) in &candidates {
                let code_slice = self.get_code_flat(vec_id as usize);

                // Try with early exit
                if let Some(dist) = hamming_distance_early_exit(&query_code.bits, code_slice, k_th_worst_distance) {
                    heap.push((dist, vec_id));
                    if heap.len() > k {
                        heap.pop();
                    }
                    if heap.len() == k {
                        k_th_worst_distance = heap.peek().unwrap().0;
                    }
                }
            }

            heap.into_sorted_vec().into_iter()
                .map(|(dist, id)| (id, 1.0 - dist as f32 / (self.code_length as f32)))
                .collect()
        }
    }

    /// Batch search with parallel processing
    pub fn batch_search_turbo(&self, queries: &[Vec<f32>], k: usize, n_candidates: usize) -> Vec<Vec<(u32, f32)>> {
        queries.par_iter().map(|q| self.search_turbo(q, k, n_candidates)).collect()
    }

    pub fn memory_bytes(&self) -> usize {
        let codes_bytes = self.codes_flat.len() * 8;
        let inv_bytes = self.inverted_index.iter().map(|v| v.len() * 4).sum::<usize>();
        let vec_bytes = self.vectors.as_ref().map(|vs| vs.iter().map(|v| v.len() * 4).sum()).unwrap_or(0);
        codes_bytes + inv_bytes + vec_bytes
    }
}

/// Brute force for ground truth
pub fn brute_force_batch(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<(u32, f32)>> {
    queries.par_iter().map(|query| {
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm < 1e-8 {
            query.to_vec()
        } else {
            query.iter().map(|x| x / query_norm).collect()
        };

        let mut results: Vec<(u32, f32)> = vectors.iter().enumerate()
            .map(|(i, v)| {
                let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if v_norm < 1e-8 { 0.0 } else {
                    dot_product(&query_normalized, v) / v_norm
                };
                (i as u32, sim)
            })
            .collect();

        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }).collect()
}
