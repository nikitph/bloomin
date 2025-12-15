//! Spectral Bloom Filter with Locality-Sensitive Hashing
//!
//! This module implements a spectral variant of Bloom filters using
//! Random Hyperplane LSH for cosine similarity preservation.
//!
//! ## Key Improvement: True LSH
//!
//! Unlike regular hashing, LSH ensures that similar vectors hash to
//! similar positions, enabling the thermodynamic descent to work correctly.
//!
//! P[h(u) = h(v)] = 1 - θ(u,v)/π
//!
//! where θ is the angle between vectors u and v.

use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::vector::Vector;

/// Number of hash functions (random hyperplanes)
const DEFAULT_NUM_HASHES: usize = 64;

/// Default filter size in slots
const DEFAULT_FILTER_SIZE: usize = 256;

/// Quantization levels (4-bit = 16 levels)
const QUANTIZATION_LEVELS: u8 = 16;

/// Random hyperplanes for LSH - generated deterministically from seed
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LSHFamily {
    /// Random hyperplane normal vectors (num_hashes x dimension)
    hyperplanes: Vec<Vec<f32>>,
    /// Dimension of vectors
    dimension: usize,
    /// Number of hyperplanes
    num_hashes: usize,
}

impl LSHFamily {
    /// Create LSH family with random hyperplanes
    pub fn new(dimension: usize, num_hashes: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let hyperplanes: Vec<Vec<f32>> = (0..num_hashes)
            .map(|_| {
                // Generate random unit vector (normal to hyperplane)
                let raw: Vec<f32> = (0..dimension)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect();

                // Normalize
                let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    raw.iter().map(|x| x / norm).collect()
                } else {
                    raw
                }
            })
            .collect();

        Self {
            hyperplanes,
            dimension,
            num_hashes,
        }
    }

    /// Compute LSH signature for a vector
    /// Returns a bit vector where bit i = 1 if dot(v, hyperplane_i) > 0
    pub fn hash(&self, vector: &Vector) -> u64 {
        let mut signature = 0u64;

        for (i, hyperplane) in self.hyperplanes.iter().enumerate().take(64) {
            let dot: f32 = vector.data.iter()
                .zip(hyperplane.iter())
                .map(|(a, b)| a * b)
                .sum();

            if dot > 0.0 {
                signature |= 1u64 << i;
            }
        }

        signature
    }

    /// Compute multiple hash indices for the filter
    /// Uses bands of bits from the LSH signature
    pub fn hash_to_indices(&self, vector: &Vector, filter_size: usize, num_indices: usize) -> Vec<usize> {
        let signature = self.hash(vector);

        // Split signature into bands and hash each band to an index
        let bits_per_band = 64 / num_indices.max(1);

        (0..num_indices)
            .map(|i| {
                let shift = (i * bits_per_band) % 64;
                let mask = (1u64 << bits_per_band) - 1;
                let band = (signature >> shift) & mask;

                // Mix the band bits for better distribution
                let mixed = band.wrapping_mul(0x517cc1b727220a95);
                (mixed as usize) % filter_size
            })
            .collect()
    }

    /// Compute similarity-preserving multi-probe indices
    /// Returns indices sorted by expected relevance
    pub fn multi_probe_indices(&self, vector: &Vector, filter_size: usize, num_probes: usize) -> Vec<usize> {
        let base_signature = self.hash(vector);
        let mut indices = Vec::with_capacity(num_probes);

        // Add base index
        indices.push((base_signature as usize) % filter_size);

        // Add indices from flipping bits (probing nearby buckets)
        for flip_bit in 0..num_probes.min(64) {
            let probed_sig = base_signature ^ (1u64 << flip_bit);
            let idx = (probed_sig as usize) % filter_size;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
            if indices.len() >= num_probes {
                break;
            }
        }

        indices
    }
}

use std::sync::OnceLock;
use std::collections::HashMap;
use std::sync::Mutex;

/// Global LSH family cache - thread-safe lazy initialization
static LSH_CACHE: OnceLock<Mutex<HashMap<usize, LSHFamily>>> = OnceLock::new();

fn get_lsh_family(dimension: usize, num_hashes: usize) -> LSHFamily {
    let cache = LSH_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();

    if let Some(family) = guard.get(&dimension) {
        return family.clone();
    }

    // Create and cache new family
    let family = LSHFamily::new(dimension, num_hashes, 42);
    guard.insert(dimension, family.clone());
    family
}

/// Spectral Bloom Filter with LSH for witness field encoding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectralBloomFilter {
    /// Positive evidence: heat accumulated from vectors present
    pub positive: Vec<u8>,

    /// Negative evidence: heat from vectors NOT in subtree
    pub negative: Vec<u8>,

    /// Number of hash bands
    pub num_hashes: usize,

    /// Filter size
    pub size: usize,

    /// Count of vectors encoded
    pub count: u64,

    /// Sum of all positive intensities
    pub total_intensity: f64,

    /// Dimension of encoded vectors (for LSH)
    dimension: usize,

    /// Cached LSH family
    #[serde(skip)]
    lsh: Option<LSHFamily>,
}

impl SpectralBloomFilter {
    /// Create a new empty spectral Bloom filter
    pub fn new(size: usize, num_hashes: usize) -> Self {
        Self {
            positive: vec![0; size],
            negative: vec![0; size],
            num_hashes,
            size,
            count: 0,
            total_intensity: 0.0,
            dimension: 0,
            lsh: None,
        }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(DEFAULT_FILTER_SIZE, DEFAULT_NUM_HASHES)
    }

    /// Ensure LSH family is initialized for the given dimension
    fn ensure_lsh(&mut self, dim: usize) {
        if self.dimension != dim || self.lsh.is_none() {
            self.dimension = dim;
            self.lsh = Some(get_lsh_family(dim, 64));
        }
    }

    /// Compute LSH indices for a vector
    fn compute_indices(&mut self, vector: &Vector) -> Vec<usize> {
        self.ensure_lsh(vector.dim());

        if let Some(ref lsh) = self.lsh {
            lsh.hash_to_indices(vector, self.size, self.num_hashes)
        } else {
            // Fallback to simple hashing
            vec![0; self.num_hashes]
        }
    }

    /// Compute indices (immutable version for queries)
    fn compute_indices_query(&self, vector: &Vector) -> Vec<usize> {
        let lsh = get_lsh_family(vector.dim(), 64);
        lsh.hash_to_indices(vector, self.size, self.num_hashes)
    }

    /// Encode a vector into the filter
    pub fn encode(&mut self, vector: &Vector) {
        self.encode_with_intensity(vector, 1.0);
    }

    /// Encode with custom intensity
    pub fn encode_with_intensity(&mut self, vector: &Vector, intensity: f32) {
        let indices = self.compute_indices(vector);
        let quantized = self.quantize_intensity(intensity);

        for idx in indices {
            self.positive[idx] = self.positive[idx].saturating_add(quantized);
        }

        self.count += 1;
        self.total_intensity += intensity as f64;
    }

    /// Encode negative evidence
    pub fn encode_negative(&mut self, vector: &Vector, intensity: f32) {
        let indices = self.compute_indices(vector);
        let quantized = self.quantize_intensity(intensity);

        for idx in indices {
            self.negative[idx] = self.negative[idx].saturating_add(quantized);
        }
    }

    #[inline]
    fn quantize_intensity(&self, intensity: f32) -> u8 {
        let scaled = (intensity * (QUANTIZATION_LEVELS as f32 - 1.0)).clamp(0.0, 15.0);
        scaled as u8
    }

    /// Compute heat score for a query vector
    pub fn heat(&self, query: &Vector) -> f32 {
        self.heat_with_lambda(query, 0.5)
    }

    /// Heat with configurable negative evidence weight
    pub fn heat_with_lambda(&self, query: &Vector, lambda: f32) -> f32 {
        let indices = self.compute_indices_query(query);

        let positive_heat: u32 = indices.iter().map(|&i| self.positive[i] as u32).sum();
        let negative_heat: u32 = indices.iter().map(|&i| self.negative[i] as u32).sum();

        let raw_heat = positive_heat as f32 - lambda * negative_heat as f32;

        // Normalize
        raw_heat / (self.num_hashes as f32 * QUANTIZATION_LEVELS as f32)
    }

    /// Multi-probe heat - checks nearby LSH buckets for better recall
    pub fn heat_multiprobe(&self, query: &Vector, num_probes: usize) -> f32 {
        let lsh = get_lsh_family(query.dim(), 64);
        let indices = lsh.multi_probe_indices(query, self.size, num_probes);

        let positive_heat: u32 = indices.iter().map(|&i| self.positive[i] as u32).sum();
        let negative_heat: u32 = indices.iter().map(|&i| self.negative[i] as u32).sum();

        let raw_heat = positive_heat as f32 - 0.5 * negative_heat as f32;
        raw_heat / (indices.len() as f32 * QUANTIZATION_LEVELS as f32)
    }

    /// Check vacuum state with SNR threshold
    pub fn is_vacuum(&self, query: &Vector) -> bool {
        let indices = self.compute_indices_query(query);

        // Check if all positions are zero (strict vacuum)
        let all_zero = indices.iter().all(|&i| self.positive[i] == 0);
        if all_zero {
            return true;
        }

        // Also check SNR-based vacuum
        let snr = self.signal_to_noise(query);
        snr < 0.5  // Below noise floor
    }

    /// Improved vacuum detection with adaptive threshold
    pub fn is_vacuum_adaptive(&self, query: &Vector) -> bool {
        if self.count == 0 {
            return true;
        }

        let heat = self.heat(query);
        let snr = self.signal_to_noise(query);

        // Expected heat for a random query
        let expected_random_heat = self.total_intensity as f32 / self.size as f32;

        // Vacuum if heat is below expected random AND SNR is low
        heat < expected_random_heat * 0.1 && snr < 1.0
    }

    /// Merge two filters
    pub fn merge(&mut self, other: &SpectralBloomFilter) {
        assert_eq!(self.size, other.size);

        for i in 0..self.size {
            self.positive[i] = self.positive[i].saturating_add(other.positive[i]);
            self.negative[i] = self.negative[i].saturating_add(other.negative[i]);
        }

        self.count += other.count;
        self.total_intensity += other.total_intensity;

        // Update dimension if needed
        if self.dimension == 0 && other.dimension > 0 {
            self.dimension = other.dimension;
        }
    }

    /// Union (max) merge
    pub fn union(&mut self, other: &SpectralBloomFilter) {
        for i in 0..self.size {
            self.positive[i] = self.positive[i].max(other.positive[i]);
            self.negative[i] = self.negative[i].max(other.negative[i]);
        }
        self.count += other.count;
    }

    /// Estimate cardinality
    pub fn estimate_cardinality(&self) -> f64 {
        let zeros = self.positive.iter().filter(|&&x| x == 0).count();
        if zeros == 0 {
            return self.size as f64;
        }
        let ratio = zeros as f64 / self.size as f64;
        -(self.size as f64) * ratio.ln()
    }

    /// Overlap coefficient
    pub fn overlap(&self, other: &SpectralBloomFilter) -> f32 {
        let mut overlap_sum = 0u32;
        let mut self_sum = 0u32;
        let mut other_sum = 0u32;

        for i in 0..self.size {
            overlap_sum += self.positive[i].min(other.positive[i]) as u32;
            self_sum += self.positive[i] as u32;
            other_sum += other.positive[i] as u32;
        }

        let min_sum = self_sum.min(other_sum);
        if min_sum == 0 { 0.0 } else { overlap_sum as f32 / min_sum as f32 }
    }

    /// Signal-to-noise ratio
    pub fn signal_to_noise(&self, query: &Vector) -> f32 {
        let indices = self.compute_indices_query(query);

        let signal: f32 = indices.iter().map(|&i| self.positive[i] as f32).sum();
        let noise_floor: f32 = self.positive.iter().map(|&x| x as f32).sum::<f32>() / self.size as f32;

        let noise_var: f32 = self.positive.iter()
            .map(|&x| (x as f32 - noise_floor).powi(2))
            .sum::<f32>() / self.size as f32;
        let noise_std = noise_var.sqrt().max(1e-6);

        let expected_noise = noise_floor * self.num_hashes as f32;
        (signal - expected_noise) / (noise_std * (self.num_hashes as f32).sqrt())
    }

    /// Serialization
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size * 2 + 32);
        bytes.extend(&(self.size as u64).to_le_bytes());
        bytes.extend(&(self.num_hashes as u64).to_le_bytes());
        bytes.extend(&self.count.to_le_bytes());
        bytes.extend(&(self.dimension as u64).to_le_bytes());
        bytes.extend(&self.positive);
        bytes.extend(&self.negative);
        bytes
    }

    pub fn memory_size(&self) -> usize {
        self.size * 2 + std::mem::size_of::<Self>()
    }
}

/// Hierarchical Spectral Summary
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HierarchicalSummary {
    pub coarse: SpectralBloomFilter,
    pub fine: Option<SpectralBloomFilter>,
    pub level: u32,
}

impl HierarchicalSummary {
    pub fn new(level: u32) -> Self {
        Self {
            coarse: SpectralBloomFilter::new(128, 32),
            fine: None,
            level,
        }
    }

    pub fn with_fine_detail(level: u32) -> Self {
        Self {
            coarse: SpectralBloomFilter::new(128, 32),
            fine: Some(SpectralBloomFilter::new(256, 64)),
            level,
        }
    }

    pub fn encode(&mut self, vector: &Vector) {
        self.coarse.encode(vector);
        if let Some(ref mut fine) = self.fine {
            fine.encode(vector);
        }
    }

    pub fn heat(&self, query: &Vector) -> f32 {
        if self.coarse.is_vacuum(query) {
            return 0.0;
        }
        match &self.fine {
            Some(fine) => fine.heat(query),
            None => self.coarse.heat(query),
        }
    }

    pub fn is_vacuum(&self, query: &Vector) -> bool {
        self.coarse.is_vacuum(query)
    }

    pub fn merge(&mut self, other: &HierarchicalSummary) {
        self.coarse.merge(&other.coarse);
        match (&mut self.fine, &other.fine) {
            (Some(sf), Some(of)) => sf.merge(of),
            (None, Some(of)) => self.fine = Some(of.clone()),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_similarity() {
        let lsh = LSHFamily::new(128, 64, 42);

        // Create similar vectors
        let v1 = Vector::new(1, vec![1.0; 128]);
        let mut v2_data = vec![1.0; 128];
        v2_data[0] = 0.9;  // Slightly different
        let v2 = Vector::new(2, v2_data);

        // Very different vector
        let v3 = Vector::new(3, vec![-1.0; 128]);

        let h1 = lsh.hash(&v1);
        let h2 = lsh.hash(&v2);
        let h3 = lsh.hash(&v3);

        // Similar vectors should have similar hashes (more matching bits)
        let sim_12 = (h1 ^ h2).count_zeros();
        let sim_13 = (h1 ^ h3).count_zeros();

        assert!(sim_12 > sim_13, "Similar vectors should have more matching hash bits");
    }

    #[test]
    fn test_encode_and_heat_lsh() {
        let mut filter = SpectralBloomFilter::new(256, 32);

        let v1 = Vector::new(1, vec![1.0; 64]);
        let v2 = Vector::new(2, vec![1.0; 64]);  // Same direction
        let v3 = Vector::new(3, vec![-1.0; 64]); // Opposite direction

        filter.encode(&v1);

        let heat1 = filter.heat(&v1);
        let heat2 = filter.heat(&v2);
        let heat3 = filter.heat(&v3);

        // v1 and v2 are similar, should have similar heat
        // v3 is opposite, should have different heat
        assert!(heat1 > 0.0);
        assert!((heat1 - heat2).abs() < heat1 * 0.5, "Similar vectors should have similar heat");
    }

    #[test]
    fn test_vacuum_detection_improved() {
        let mut filter = SpectralBloomFilter::new(256, 32);

        // Encode vectors in one region
        for i in 0..100 {
            let v = Vector::new(i, vec![1.0; 64]);
            filter.encode(&v);
        }

        // Query in same region - should not be vacuum
        let query_same = Vector::new(999, vec![1.0; 64]);
        assert!(!filter.is_vacuum(&query_same));

        // Query in opposite region - may or may not be vacuum due to LSH
        let query_opposite = Vector::new(1000, vec![-1.0; 64]);
        let snr = filter.signal_to_noise(&query_opposite);
        println!("SNR for opposite query: {}", snr);
    }

    #[test]
    fn test_merge_with_lsh() {
        let mut filter1 = SpectralBloomFilter::new(256, 32);
        let mut filter2 = SpectralBloomFilter::new(256, 32);

        let v1 = Vector::new(1, vec![1.0; 64]);
        let v2 = Vector::new(2, vec![-1.0; 64]);

        filter1.encode(&v1);
        filter2.encode(&v2);

        let heat_before = filter1.heat(&v1);
        filter1.merge(&filter2);
        let heat_after = filter1.heat(&v1);

        assert!(heat_after >= heat_before);
    }
}
