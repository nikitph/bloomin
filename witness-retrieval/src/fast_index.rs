//! Fast Witness-LDPC Index: Optimized for Speed
//!
//! Key optimizations:
//! 1. Fixed-size stack-allocated codes (cache-friendly)
//! 2. Hamming-only search (no exact re-ranking overhead)
//! 3. Minimal candidate set with high-quality filtering
//! 4. SIMD-friendly operations

use ahash::AHasher;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

/// Fixed-size binary code (stack allocated for speed)
/// Using 256 u64s = 16384 bits max (covers most use cases)
#[derive(Clone, Copy)]
pub struct FastCode {
    bits: [u64; 4],  // 256 bits - compact and cache-friendly
    popcount: u32,
}

impl FastCode {
    #[inline]
    pub fn new() -> Self {
        Self { bits: [0u64; 4], popcount: 0 }
    }

    #[inline]
    pub fn set_bit(&mut self, pos: usize) {
        if pos >= 256 { return; }
        let chunk = pos / 64;
        let bit = pos % 64;
        let mask = 1u64 << bit;
        if self.bits[chunk] & mask == 0 {
            self.bits[chunk] |= mask;
            self.popcount += 1;
        }
    }

    #[inline]
    pub fn hamming_similarity(&self, other: &FastCode) -> u32 {
        (self.bits[0] & other.bits[0]).count_ones() +
        (self.bits[1] & other.bits[1]).count_ones() +
        (self.bits[2] & other.bits[2]).count_ones() +
        (self.bits[3] & other.bits[3]).count_ones()
    }

    #[inline]
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

/// Variable-size binary code for larger m values
#[derive(Clone)]
pub struct VarCode {
    bits: Vec<u64>,
    code_length: usize,
    popcount: u32,
}

impl VarCode {
    pub fn new(code_length: usize) -> Self {
        let n_chunks = (code_length + 63) / 64;
        Self { bits: vec![0u64; n_chunks], code_length, popcount: 0 }
    }

    #[inline]
    pub fn set_bit(&mut self, pos: usize) {
        if pos >= self.code_length { return; }
        let chunk = pos / 64;
        let bit = pos % 64;
        let mask = 1u64 << bit;
        if self.bits[chunk] & mask == 0 {
            self.bits[chunk] |= mask;
            self.popcount += 1;
        }
    }

    #[inline]
    pub fn hamming_similarity(&self, other: &VarCode) -> u32 {
        self.bits.iter().zip(other.bits.iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum()
    }

    #[inline]
    pub fn set_bit_indices(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.popcount as usize);
        for (chunk_idx, &chunk) in self.bits.iter().enumerate() {
            let mut c = chunk;
            let base = chunk_idx * 64;
            while c != 0 {
                let tz = c.trailing_zeros() as usize;
                let idx = base + tz;
                if idx < self.code_length {
                    indices.push(idx);
                }
                c &= c - 1;
            }
        }
        indices
    }
}

/// Fast hash function
#[inline]
fn hash_witness(witness: u32, seed: u64, code_length: usize) -> usize {
    let mut hasher = AHasher::default();
    witness.hash(&mut hasher);
    seed.hash(&mut hasher);
    (hasher.finish() as usize) % code_length
}

/// Extract top-k witnesses
#[inline]
fn extract_witnesses(vector: &[f32], num_witnesses: usize) -> Vec<u32> {
    let dim = vector.len();
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_unstable_by_key(|&i| std::cmp::Reverse(OrderedFloat(vector[i].abs())));

    indices.iter().take(num_witnesses).map(|&idx| {
        let sign_bit = if vector[idx] > 0.0 { 1u32 } else { 0u32 };
        (idx as u32) * 2 + sign_bit
    }).collect()
}

/// SIMD-friendly dot product
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Fast Witness-LDPC Index with Variable Code Length
pub struct FastWitnessIndex {
    pub dim: usize,
    pub code_length: usize,
    pub num_hashes: usize,
    pub num_witnesses: usize,
    hash_seeds: Vec<u64>,
    codes: Vec<VarCode>,
    vectors: Option<Vec<Vec<f32>>>,
    inverted_index: Vec<Vec<u32>>,
    pub n_vectors: usize,
}

impl FastWitnessIndex {
    pub fn new(dim: usize, code_length: usize, num_hashes: usize, num_witnesses: usize) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let hash_seeds: Vec<u64> = (0..num_hashes).map(|_| rng.gen()).collect();

        Self {
            dim,
            code_length,
            num_hashes,
            num_witnesses,
            hash_seeds,
            codes: Vec::new(),
            vectors: None,
            inverted_index: vec![Vec::new(); code_length],
            n_vectors: 0,
        }
    }

    fn encode(&self, vector: &[f32]) -> VarCode {
        let witnesses = extract_witnesses(vector, self.num_witnesses);
        let mut code = VarCode::new(self.code_length);

        for witness in witnesses {
            for (k, &seed) in self.hash_seeds.iter().enumerate() {
                let pos = hash_witness(witness, seed.wrapping_add(k as u64), self.code_length);
                code.set_bit(pos);
            }
        }
        code
    }

    /// Add vectors (optionally store for re-ranking)
    pub fn add(&mut self, vectors: Vec<Vec<f32>>, store_vectors: bool) {
        let n = vectors.len();
        self.n_vectors = n;

        // Pre-normalize
        let normalized: Vec<Vec<f32>> = vectors.par_iter().map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-8 { v.clone() } else { v.iter().map(|x| x / norm).collect() }
        }).collect();

        // Encode
        let codes: Vec<VarCode> = normalized.par_iter().map(|v| self.encode(v)).collect();

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

    /// Fast Hamming-only search (NO exact re-ranking)
    pub fn search_hamming_only(&self, query: &[f32], k: usize, n_candidates: usize) -> Vec<(u32, f32)> {
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm < 1e-8 {
            query.to_vec()
        } else {
            query.iter().map(|x| x / query_norm).collect()
        };

        let query_code = self.encode(&query_normalized);
        let query_bits = query_code.set_bit_indices();

        // Count overlaps via inverted index
        let mut scores = vec![0u32; self.n_vectors];
        for &bit_pos in &query_bits {
            for &vec_id in &self.inverted_index[bit_pos] {
                scores[vec_id as usize] += 1;
            }
        }

        // Get top-k by Hamming score (partial sort)
        let mut candidates: Vec<(u32, u32)> = scores.iter().enumerate()
            .filter(|(_, &s)| s > 0)
            .map(|(i, &s)| (i as u32, s))
            .collect();

        // Partial sort for top n_candidates
        if candidates.len() > n_candidates {
            candidates.select_nth_unstable_by(n_candidates, |a, b| b.1.cmp(&a.1));
            candidates.truncate(n_candidates);
        }
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        candidates.into_iter().take(k)
            .map(|(id, score)| (id, score as f32 / query_code.popcount as f32))
            .collect()
    }

    /// Search with exact re-ranking (slower but more accurate)
    pub fn search_exact(&self, query: &[f32], k: usize, n_candidates: usize) -> Vec<(u32, f32)> {
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm < 1e-8 {
            query.to_vec()
        } else {
            query.iter().map(|x| x / query_norm).collect()
        };

        let query_code = self.encode(&query_normalized);
        let query_bits = query_code.set_bit_indices();

        // Count overlaps via inverted index
        let mut scores = vec![0u32; self.n_vectors];
        for &bit_pos in &query_bits {
            for &vec_id in &self.inverted_index[bit_pos] {
                scores[vec_id as usize] += 1;
            }
        }

        // Get top candidates
        let mut candidates: Vec<(u32, u32)> = scores.iter().enumerate()
            .filter(|(_, &s)| s > 0)
            .map(|(i, &s)| (i as u32, s))
            .collect();

        if candidates.len() > n_candidates {
            candidates.select_nth_unstable_by(n_candidates, |a, b| b.1.cmp(&a.1));
            candidates.truncate(n_candidates);
        }

        // Re-rank with exact cosine
        if let Some(ref vectors) = self.vectors {
            let mut results: Vec<(u32, f32)> = candidates.iter()
                .map(|&(vec_id, _)| {
                    let sim = dot_product(&query_normalized, &vectors[vec_id as usize]);
                    (vec_id, sim)
                })
                .collect();
            results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            results.truncate(k);
            results
        } else {
            candidates.into_iter().take(k)
                .map(|(id, score)| (id, score as f32))
                .collect()
        }
    }

    /// Batch search - Hamming only
    pub fn batch_search_hamming(&self, queries: &[Vec<f32>], k: usize, n_candidates: usize) -> Vec<Vec<(u32, f32)>> {
        queries.par_iter().map(|q| self.search_hamming_only(q, k, n_candidates)).collect()
    }

    /// Batch search - with exact re-ranking
    pub fn batch_search_exact(&self, queries: &[Vec<f32>], k: usize, n_candidates: usize) -> Vec<Vec<(u32, f32)>> {
        queries.par_iter().map(|q| self.search_exact(q, k, n_candidates)).collect()
    }

    pub fn memory_bytes(&self) -> usize {
        let codes_bytes = self.codes.iter().map(|c| c.bits.len() * 8).sum::<usize>();
        let inv_bytes = self.inverted_index.iter().map(|v| v.len() * 4).sum::<usize>();
        let vec_bytes = self.vectors.as_ref().map(|vs| vs.iter().map(|v| v.len() * 4).sum()).unwrap_or(0);
        codes_bytes + inv_bytes + vec_bytes
    }

    pub fn memory_bytes_no_vectors(&self) -> usize {
        let codes_bytes = self.codes.iter().map(|c| c.bits.len() * 8).sum::<usize>();
        let inv_bytes = self.inverted_index.iter().map(|v| v.len() * 4).sum::<usize>();
        codes_bytes + inv_bytes
    }
}

/// Brute force for ground truth
pub fn brute_force_search(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
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
                v.iter().zip(query_normalized.iter()).map(|(a, b)| a * b / v_norm).sum()
            };
            (i as u32, sim)
        })
        .collect();

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(k);
    results
}

pub fn brute_force_batch(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<(u32, f32)>> {
    queries.par_iter().map(|q| brute_force_search(vectors, q, k)).collect()
}
