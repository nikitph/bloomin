//! Hierarchical Witness-LDPC v2: Production System
//!
//! Implements ALL the theoretical insights:
//! 1. Adaptive m ∝ log(N) at each level
//! 2. IDF-based witness extraction (prioritize rare dimensions)
//! 3. Sparse witnesses at coarse level (fewer, higher quality)
//! 4. Proper candidate pool scaling
//!
//! Architecture:
//! - Coarse: 256 bits, K=2, w=16 (fast filter, N → 1%)
//! - Medium: adaptive m, K=3, w=32 (1% → 0.5%)
//! - Fine: adaptive m, K=6, w=96 (0.5% → top-k)

use ahash::AHasher;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Fast hash function
#[inline]
fn hash_witness(witness: u32, seed: u64, code_length: usize) -> usize {
    let mut hasher = AHasher::default();
    witness.hash(&mut hasher);
    seed.hash(&mut hasher);
    (hasher.finish() as usize) % code_length
}

/// SIMD-friendly dot product
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalize vector
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Next power of 2
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 { return 1; }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// IDF scores for witness selection
pub struct IDFScores {
    scores: Vec<f32>,
}

impl IDFScores {
    /// Compute IDF scores: IDF[i] = log(N / (1 + df[i]))
    pub fn compute(vectors: &[Vec<f32>], threshold: f32) -> Self {
        let n = vectors.len();
        let dim = vectors[0].len();

        // Count document frequency per dimension
        let mut doc_freq = vec![0usize; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                if val.abs() > threshold {
                    doc_freq[i] += 1;
                }
            }
        }

        // IDF = log(N / (1 + df))
        let scores: Vec<f32> = doc_freq
            .iter()
            .map(|&df| ((n as f32) / (1.0 + df as f32)).ln())
            .collect();

        Self { scores }
    }

    /// Extract witnesses weighted by IDF (prioritize rare dimensions)
    pub fn extract_witnesses(&self, vector: &[f32], k: usize, use_signs: bool) -> Vec<u32> {
        let mut weighted_scores: Vec<(usize, f32)> = vector
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.abs() * self.scores[i]))
            .collect();

        weighted_scores.sort_unstable_by_key(|(_, score)| std::cmp::Reverse(OrderedFloat(*score)));

        let mut witnesses = Vec::with_capacity(k);
        for &(idx, _) in weighted_scores.iter().take(k) {
            if use_signs {
                let sign_bit = if vector[idx] > 0.0 { 1 } else { 0 };
                witnesses.push((idx as u32) * 2 + sign_bit);
            } else {
                witnesses.push(idx as u32);
            }
        }
        witnesses
    }
}

/// Standard witness extraction (without IDF)
fn extract_witnesses_standard(vector: &[f32], k: usize, use_signs: bool) -> Vec<u32> {
    let dim = vector.len();
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_unstable_by_key(|&i| std::cmp::Reverse(OrderedFloat(vector[i].abs())));

    let mut witnesses = Vec::with_capacity(k);
    for &idx in indices.iter().take(k) {
        if use_signs {
            let sign_bit = if vector[idx] > 0.0 { 1 } else { 0 };
            witnesses.push((idx as u32) * 2 + sign_bit);
        } else {
            witnesses.push(idx as u32);
        }
    }
    witnesses
}

/// Binary code with variable length
#[derive(Clone)]
pub struct BinaryCode {
    pub bits: Vec<u64>,
    pub code_length: usize,
    pub popcount: u32,
}

impl BinaryCode {
    pub fn new(code_length: usize) -> Self {
        let n_chunks = (code_length + 63) / 64;
        Self {
            bits: vec![0u64; n_chunks],
            code_length,
            popcount: 0,
        }
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
    pub fn hamming_similarity(&self, other: &BinaryCode) -> u32 {
        self.bits.iter().zip(other.bits.iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum()
    }

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

/// Single level encoder
struct LevelEncoder {
    m: usize,
    k: usize,
    witnesses: usize,
    seeds: Vec<u64>,
    use_signs: bool,
}

impl LevelEncoder {
    fn new(m: usize, k: usize, witnesses: usize, seed_base: u64) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed_base);
        let seeds: Vec<u64> = (0..k).map(|_| rng.gen()).collect();

        Self { m, k, witnesses, seeds, use_signs: true }
    }

    fn encode(&self, witness_list: &[u32]) -> BinaryCode {
        let mut code = BinaryCode::new(self.m);
        for &witness in witness_list.iter().take(self.witnesses) {
            for (k, &seed) in self.seeds.iter().enumerate() {
                let pos = hash_witness(witness, seed.wrapping_add(k as u64), self.m);
                code.set_bit(pos);
            }
        }
        code
    }
}

/// Compute adaptive m for coarse level
/// Must scale with full N to avoid complete hash collisions
fn compute_coarse_m(n: usize) -> usize {
    // m ∝ log(N), base: m=512 at N=50K
    let m = 512.0 * ((n as f64).ln() / (50_000_f64).ln());
    next_power_of_two(m.ceil() as usize).max(512).min(4096)
}

/// Compute adaptive m for medium level
/// Needs to discriminate among ~5% of dataset
fn compute_medium_m(n: usize) -> usize {
    // m ∝ log(N), base: m=2048 at N=50K
    let m = 2048.0 * ((n as f64).ln() / (50_000_f64).ln());
    next_power_of_two(m.ceil() as usize).max(1024).min(16384)
}

/// Compute adaptive m for fine level
/// Needs to discriminate among ~1% of dataset
fn compute_fine_m(n: usize) -> usize {
    // m ∝ log(N), base: m=8192 at N=50K
    let m = 8192.0 * ((n as f64).ln() / (50_000_f64).ln());
    next_power_of_two(m.ceil() as usize).max(4096).min(65536)
}

/// Compute adaptive candidate counts
fn compute_candidates(n: usize) -> (usize, usize) {
    // Coarse: ~5% of dataset, min 2000
    let coarse = ((n as f64 * 0.05) as usize).max(2000).min(25000);
    // Medium: ~2% of dataset, min 1000
    let medium = ((n as f64 * 0.02) as usize).max(1000).min(10000);
    (coarse, medium)
}

/// Production Hierarchical Witness-LDPC v2
pub struct HierarchicalV2 {
    pub dim: usize,
    pub n_vectors: usize,

    // IDF scores
    idf_scores: Option<IDFScores>,

    // Level encoders
    coarse: LevelEncoder,
    medium: LevelEncoder,
    fine: LevelEncoder,

    // Encoded vectors
    coarse_codes: Vec<BinaryCode>,
    medium_codes: Vec<BinaryCode>,
    fine_codes: Vec<BinaryCode>,

    // Inverted indices (coarse and medium only)
    coarse_inverted: Vec<Vec<u32>>,
    medium_inverted: Vec<Vec<u32>>,

    // Candidate counts
    coarse_candidates: usize,
    medium_candidates: usize,

    // Original vectors (pre-normalized)
    vectors: Option<Vec<Vec<f32>>>,
}

impl HierarchicalV2 {
    /// Create new hierarchical index with adaptive parameters
    pub fn new(dim: usize, n: usize, use_idf: bool) -> Self {
        // Compute adaptive m values at ALL levels
        let coarse_m = compute_coarse_m(n);
        let medium_m = compute_medium_m(n);
        let fine_m = compute_fine_m(n);
        let (coarse_candidates, medium_candidates) = compute_candidates(n);

        println!("HierarchicalV2 Configuration:");
        println!("  Dataset size: {}", n);
        println!("  Coarse: m={}, K=2, w=16 (adaptive)", coarse_m);
        println!("  Medium: m={}, K=4, w=48 (adaptive)", medium_m);
        println!("  Fine:   m={}, K=6, w=96 (adaptive)", fine_m);
        println!("  Candidates: coarse={}, medium={}", coarse_candidates, medium_candidates);

        Self {
            dim,
            n_vectors: 0,
            idf_scores: None,
            coarse: LevelEncoder::new(coarse_m, 2, 16, 42),
            medium: LevelEncoder::new(medium_m, 4, 48, 123),  // Increased witnesses and hashes
            fine: LevelEncoder::new(fine_m, 6, 96, 456),
            coarse_codes: Vec::new(),
            medium_codes: Vec::new(),
            fine_codes: Vec::new(),
            coarse_inverted: Vec::new(),
            medium_inverted: Vec::new(),
            coarse_candidates,
            medium_candidates,
            vectors: None,
        }
    }

    /// Add vectors to the index
    pub fn add(&mut self, vectors: Vec<Vec<f32>>, store_vectors: bool, use_idf: bool) {
        let n = vectors.len();
        self.n_vectors = n;

        println!("Building HierarchicalV2 index...");

        // Pre-normalize vectors
        println!("  Normalizing vectors...");
        let normalized: Vec<Vec<f32>> = vectors.par_iter().map(|v| normalize(v)).collect();

        // Compute IDF if requested
        if use_idf {
            println!("  Computing IDF scores...");
            self.idf_scores = Some(IDFScores::compute(&normalized, 0.1));
        }

        // Extract witnesses and encode all three levels in parallel
        println!("  Encoding vectors (3 levels)...");
        let results: Vec<(BinaryCode, BinaryCode, BinaryCode)> = normalized
            .par_iter()
            .map(|v| {
                // Extract witnesses (with or without IDF)
                let witnesses = if let Some(ref idf) = self.idf_scores {
                    idf.extract_witnesses(v, 96, true) // max witnesses for fine level
                } else {
                    extract_witnesses_standard(v, 96, true)
                };

                let coarse = self.coarse.encode(&witnesses);
                let medium = self.medium.encode(&witnesses);
                let fine = self.fine.encode(&witnesses);

                (coarse, medium, fine)
            })
            .collect();

        // Unpack results
        let mut coarse_codes = Vec::with_capacity(n);
        let mut medium_codes = Vec::with_capacity(n);
        let mut fine_codes = Vec::with_capacity(n);

        for (c, m, f) in results {
            coarse_codes.push(c);
            medium_codes.push(m);
            fine_codes.push(f);
        }

        // Build inverted indices
        println!("  Building inverted indices...");

        self.coarse_inverted = vec![Vec::new(); self.coarse.m];
        for (vec_id, code) in coarse_codes.iter().enumerate() {
            for bit_pos in code.set_bit_indices() {
                self.coarse_inverted[bit_pos].push(vec_id as u32);
            }
        }

        self.medium_inverted = vec![Vec::new(); self.medium.m];
        for (vec_id, code) in medium_codes.iter().enumerate() {
            for bit_pos in code.set_bit_indices() {
                self.medium_inverted[bit_pos].push(vec_id as u32);
            }
        }

        self.coarse_codes = coarse_codes;
        self.medium_codes = medium_codes;
        self.fine_codes = fine_codes;

        if store_vectors {
            self.vectors = Some(normalized);
        }

        // Statistics
        let avg_coarse: f64 = self.coarse_codes.iter().map(|c| c.popcount as f64).sum::<f64>() / n as f64;
        let avg_medium: f64 = self.medium_codes.iter().map(|c| c.popcount as f64).sum::<f64>() / n as f64;
        let avg_fine: f64 = self.fine_codes.iter().map(|c| c.popcount as f64).sum::<f64>() / n as f64;

        println!("Index built:");
        println!("  Coarse: {:.1} bits/vector (density={:.1}%)", avg_coarse, avg_coarse / self.coarse.m as f64 * 100.0);
        println!("  Medium: {:.1} bits/vector (density={:.1}%)", avg_medium, avg_medium / self.medium.m as f64 * 100.0);
        println!("  Fine:   {:.1} bits/vector (density={:.1}%)", avg_fine, avg_fine / self.fine.m as f64 * 100.0);
        println!("  Memory: {:.1} MB", self.memory_bytes() as f64 / 1024.0 / 1024.0);
    }

    /// 3-level cascaded search
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let query_normalized = normalize(query);

        // Extract witnesses
        let witnesses = if let Some(ref idf) = self.idf_scores {
            idf.extract_witnesses(&query_normalized, 96, true)
        } else {
            extract_witnesses_standard(&query_normalized, 96, true)
        };

        // === Stage 1: Coarse filter (N → coarse_candidates) ===
        let query_coarse = self.coarse.encode(&witnesses);
        let query_coarse_bits = query_coarse.set_bit_indices();

        let mut coarse_scores: HashMap<u32, u32> = HashMap::with_capacity(self.coarse_candidates * 2);
        for &bit_pos in &query_coarse_bits {
            if bit_pos < self.coarse_inverted.len() {
                for &vec_id in &self.coarse_inverted[bit_pos] {
                    *coarse_scores.entry(vec_id).or_insert(0) += 1;
                }
            }
        }

        if coarse_scores.is_empty() {
            return Vec::new();
        }

        let mut stage1: Vec<(u32, u32)> = coarse_scores.into_iter().collect();
        stage1.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        stage1.truncate(self.coarse_candidates);

        // === Stage 2: Medium re-ranking (coarse_candidates → medium_candidates) ===
        let query_medium = self.medium.encode(&witnesses);

        let mut stage2: Vec<(u32, u32)> = stage1
            .iter()
            .map(|&(vec_id, _)| {
                let sim = query_medium.hamming_similarity(&self.medium_codes[vec_id as usize]);
                (vec_id, sim)
            })
            .collect();

        stage2.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        stage2.truncate(self.medium_candidates);

        // === Stage 3: Fine re-ranking (medium_candidates → k*3) ===
        let query_fine = self.fine.encode(&witnesses);

        let mut stage3: Vec<(u32, u32)> = stage2
            .iter()
            .map(|&(vec_id, _)| {
                let sim = query_fine.hamming_similarity(&self.fine_codes[vec_id as usize]);
                (vec_id, sim)
            })
            .collect();

        stage3.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // === Stage 4: Exact re-ranking with original vectors ===
        if let Some(ref vectors) = self.vectors {
            let top_fine: Vec<u32> = stage3.iter().take(k * 3).map(|(id, _)| *id).collect();

            let mut exact: Vec<(u32, f32)> = top_fine
                .iter()
                .map(|&vec_id| {
                    let sim = dot_product(&query_normalized, &vectors[vec_id as usize]);
                    (vec_id, sim)
                })
                .collect();

            exact.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            exact.truncate(k);
            exact
        } else {
            stage3.into_iter().take(k).map(|(id, s)| (id, s as f32)).collect()
        }
    }

    /// Batch search (parallelized)
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(u32, f32)>> {
        queries.par_iter().map(|q| self.search(q, k)).collect()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let coarse_bytes = self.coarse_codes.iter().map(|c| c.bits.len() * 8).sum::<usize>();
        let medium_bytes = self.medium_codes.iter().map(|c| c.bits.len() * 8).sum::<usize>();
        let fine_bytes = self.fine_codes.iter().map(|c| c.bits.len() * 8).sum::<usize>();
        let coarse_inv = self.coarse_inverted.iter().map(|v| v.len() * 4).sum::<usize>();
        let medium_inv = self.medium_inverted.iter().map(|v| v.len() * 4).sum::<usize>();
        let vectors_bytes = self.vectors.as_ref()
            .map(|vs| vs.iter().map(|v| v.len() * 4).sum())
            .unwrap_or(0);

        coarse_bytes + medium_bytes + fine_bytes + coarse_inv + medium_inv + vectors_bytes
    }
}

/// Brute force search for ground truth
pub fn brute_force_search(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let query_normalized = normalize(query);

    let mut results: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let v_norm = normalize(v);
            let sim = dot_product(&query_normalized, &v_norm);
            (i as u32, sim)
        })
        .collect();

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(k);
    results
}

/// Parallel brute force
pub fn brute_force_batch_search(vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<(u32, f32)>> {
    queries.par_iter().map(|q| brute_force_search(vectors, q, k)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_hierarchical_v2() {
        let mut rng = rand::thread_rng();
        let dim = 768;
        let n = 10000;

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();

        let mut index = HierarchicalV2::new(dim, n, true);
        index.add(vectors.clone(), true, true);

        let results = index.search(&vectors[0], 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Should find itself
    }
}
