//! Witness-LDPC Codes for Fast Similarity Search
//!
//! Theory: High-dimensional vectors have geometric structure that can be captured
//! by "witnesses" - the most distinctive dimensions. Using expander-graph-like
//! hashing provides good distance preservation with compact binary codes.

pub mod fast_index;
pub mod hierarchical;
pub mod hierarchical_v2;
pub mod production;
pub mod turbo_index;

use ahash::AHasher;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// A single vector's binary code, bit-packed into u64 chunks
#[derive(Clone)]
pub struct BinaryCode {
    /// Bit-packed code. code_length bits stored in ceil(code_length/64) u64s
    pub bits: Vec<u64>,
    /// Number of bits set (popcount)
    pub popcount: u32,
}

impl BinaryCode {
    pub fn new(code_length: usize) -> Self {
        let n_chunks = (code_length + 63) / 64;
        Self {
            bits: vec![0u64; n_chunks],
            popcount: 0,
        }
    }

    #[inline]
    pub fn set_bit(&mut self, pos: usize) {
        let chunk = pos / 64;
        let bit = pos % 64;
        let mask = 1u64 << bit;
        if self.bits[chunk] & mask == 0 {
            self.bits[chunk] |= mask;
            self.popcount += 1;
        }
    }

    #[inline]
    pub fn get_bit(&self, pos: usize) -> bool {
        let chunk = pos / 64;
        let bit = pos % 64;
        (self.bits[chunk] >> bit) & 1 == 1
    }

    /// Hamming similarity (number of matching 1-bits)
    #[inline]
    pub fn hamming_similarity(&self, other: &BinaryCode) -> u32 {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum()
    }

    /// Get indices of set bits
    pub fn set_bit_indices(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.popcount as usize);
        for (chunk_idx, &chunk) in self.bits.iter().enumerate() {
            let mut c = chunk;
            let base = chunk_idx * 64;
            while c != 0 {
                let tz = c.trailing_zeros() as usize;
                indices.push(base + tz);
                c &= c - 1; // Clear lowest set bit
            }
        }
        indices
    }
}

/// Fast hash function for witness encoding
#[inline]
fn hash_witness(witness: u32, seed: u64, code_length: usize) -> usize {
    let mut hasher = AHasher::default();
    witness.hash(&mut hasher);
    seed.hash(&mut hasher);
    (hasher.finish() as usize) % code_length
}

/// Extract top-k witnesses (dimensions with largest absolute values)
fn extract_witnesses(vector: &[f32], num_witnesses: usize, use_signs: bool) -> Vec<u32> {
    let dim = vector.len();

    // Get indices sorted by absolute value (descending)
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_unstable_by_key(|&i| std::cmp::Reverse(OrderedFloat(vector[i].abs())));

    let mut witnesses = Vec::with_capacity(num_witnesses);
    for &idx in indices.iter().take(num_witnesses) {
        if use_signs {
            let sign_bit = if vector[idx] > 0.0 { 1 } else { 0 };
            witnesses.push((idx as u32) * 2 + sign_bit);
        } else {
            witnesses.push(idx as u32);
        }
    }
    witnesses
}

/// SIMD-friendly dot product
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Witness-LDPC Index
pub struct WitnessLDPCIndex {
    /// Vector dimension
    pub dim: usize,
    /// Binary code length
    pub code_length: usize,
    /// Number of hash functions per witness
    pub num_hashes: usize,
    /// Number of witnesses per vector
    pub num_witnesses: usize,
    /// Whether to encode sign information
    pub use_signs: bool,
    /// Hash seeds
    hash_seeds: Vec<u64>,
    /// Encoded vectors
    codes: Vec<BinaryCode>,
    /// Original vectors (PRE-NORMALIZED for fast cosine sim)
    vectors: Option<Vec<Vec<f32>>>,
    /// Inverted index: bit position -> vector IDs
    inverted_index: Vec<Vec<u32>>,
    /// Number of indexed vectors
    pub n_vectors: usize,
}

impl WitnessLDPCIndex {
    pub fn new(
        dim: usize,
        code_length: usize,
        num_hashes: usize,
        num_witnesses: usize,
        use_signs: bool,
    ) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let hash_seeds: Vec<u64> = (0..num_hashes).map(|_| rng.gen()).collect();

        Self {
            dim,
            code_length,
            num_hashes,
            num_witnesses,
            use_signs,
            hash_seeds,
            codes: Vec::new(),
            vectors: None,
            inverted_index: vec![Vec::new(); code_length],
            n_vectors: 0,
        }
    }

    /// Encode a single vector to binary code
    pub fn encode(&self, vector: &[f32]) -> BinaryCode {
        let witnesses = extract_witnesses(vector, self.num_witnesses, self.use_signs);

        let mut code = BinaryCode::new(self.code_length);

        for witness in witnesses {
            for (k, &seed) in self.hash_seeds.iter().enumerate() {
                let pos = hash_witness(witness, seed.wrapping_add(k as u64), self.code_length);
                code.set_bit(pos);
            }
        }

        code
    }

    /// Add vectors to the index
    pub fn add(&mut self, vectors: Vec<Vec<f32>>, store_vectors: bool) {
        let n = vectors.len();
        self.n_vectors = n;

        println!("Encoding {} vectors (dim={}, code_length={})...", n, self.dim, self.code_length);

        // Pre-normalize vectors for fast cosine similarity
        let normalized_vectors: Vec<Vec<f32>> = vectors
            .par_iter()
            .map(|v| {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-8 {
                    v.clone()
                } else {
                    v.iter().map(|x| x / norm).collect()
                }
            })
            .collect();

        // Encode in parallel
        let codes: Vec<BinaryCode> = normalized_vectors
            .par_iter()
            .map(|v| self.encode(v))
            .collect();

        // Build inverted index
        println!("Building inverted index...");
        self.inverted_index = vec![Vec::new(); self.code_length];

        for (vec_id, code) in codes.iter().enumerate() {
            for bit_pos in code.set_bit_indices() {
                self.inverted_index[bit_pos].push(vec_id as u32);
            }
        }

        self.codes = codes;

        if store_vectors {
            // Store PRE-NORMALIZED vectors
            self.vectors = Some(normalized_vectors);
        }

        // Statistics
        let avg_bits: f64 = self.codes.iter().map(|c| c.popcount as f64).sum::<f64>() / n as f64;
        let active_bits = self.inverted_index.iter().filter(|v| !v.is_empty()).count();

        println!("Index built:");
        println!("  - {} vectors", n);
        println!("  - {} active bits", active_bits);
        println!("  - {:.1} bits per vector (avg)", avg_bits);
    }

    /// Search for k nearest neighbors (optimized)
    pub fn search(&self, query: &[f32], k: usize, n_candidates: usize) -> Vec<(u32, f32)> {
        // Pre-normalize query ONCE
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm < 1e-8 {
            query.to_vec()
        } else {
            query.iter().map(|x| x / query_norm).collect()
        };

        let query_code = self.encode(&query_normalized);
        let query_bits = query_code.set_bit_indices();

        // Fast candidate retrieval via inverted index
        let mut candidate_scores: HashMap<u32, u32> = HashMap::with_capacity(n_candidates * 2);

        for &bit_pos in &query_bits {
            for &vec_id in &self.inverted_index[bit_pos] {
                *candidate_scores.entry(vec_id).or_insert(0) += 1;
            }
        }

        if candidate_scores.is_empty() {
            return Vec::new();
        }

        // Get top candidates by Hamming similarity
        let mut candidates: Vec<(u32, u32)> = candidate_scores.into_iter().collect();
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        candidates.truncate(n_candidates);

        // Re-rank using exact cosine similarity (vectors are pre-normalized!)
        if let Some(ref vectors) = self.vectors {
            let mut results: Vec<(u32, f32)> = candidates
                .iter()
                .map(|&(vec_id, _)| {
                    let sim = dot_product(&query_normalized, &vectors[vec_id as usize]);
                    (vec_id, sim)
                })
                .collect();

            results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            results.truncate(k);
            results
        } else {
            // Return Hamming similarity scores
            candidates
                .into_iter()
                .take(k)
                .map(|(id, score)| (id, score as f32 / query_bits.len() as f32))
                .collect()
        }
    }

    /// Batch search (parallelized)
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize, n_candidates: usize) -> Vec<Vec<(u32, f32)>> {
        queries
            .par_iter()
            .map(|q| self.search(q, k, n_candidates))
            .collect()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let codes_bytes = self.codes.iter().map(|c| c.bits.len() * 8).sum::<usize>();
        let inv_idx_bytes = self.inverted_index.iter().map(|v| v.len() * 4).sum::<usize>();
        let vectors_bytes = self.vectors.as_ref().map(|vs| vs.iter().map(|v| v.len() * 4).sum()).unwrap_or(0);

        codes_bytes + inv_idx_bytes + vectors_bytes
    }
}

/// Brute force exact search (for ground truth) - also pre-normalizes
pub fn brute_force_search(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let query_normalized: Vec<f32> = if query_norm < 1e-8 {
        query.to_vec()
    } else {
        query.iter().map(|x| x / query_norm).collect()
    };

    let mut results: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let sim = if v_norm < 1e-8 {
                0.0
            } else {
                v.iter().zip(query_normalized.iter()).map(|(a, b)| a * b / v_norm).sum()
            };
            (i as u32, sim)
        })
        .collect();

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(k);
    results
}

/// Parallel brute force for multiple queries
pub fn brute_force_batch_search(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<(u32, f32)>> {
    queries
        .par_iter()
        .map(|q| brute_force_search(vectors, q, k))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_binary_code() {
        let mut code = BinaryCode::new(128);
        code.set_bit(0);
        code.set_bit(63);
        code.set_bit(64);
        code.set_bit(127);

        assert!(code.get_bit(0));
        assert!(code.get_bit(63));
        assert!(code.get_bit(64));
        assert!(code.get_bit(127));
        assert!(!code.get_bit(1));
        assert_eq!(code.popcount, 4);
    }

    #[test]
    fn test_witness_extraction() {
        let vector = vec![0.1, -0.5, 0.3, -0.8, 0.2];
        let witnesses = extract_witnesses(&vector, 3, true);
        assert_eq!(witnesses.len(), 3);
    }

    #[test]
    fn test_index() {
        let mut rng = rand::thread_rng();
        let dim = 128;
        let n = 1000;

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();

        let mut index = WitnessLDPCIndex::new(dim, 512, 4, 32, true);
        index.add(vectors.clone(), true);

        let results = index.search(&vectors[0], 10, 100);
        assert!(!results.is_empty());
    }
}
