//! REWA Quantizer: Projects dense vectors into binary signatures
//!
//! Implements Theorem 4.1 from the REWA framework:
//! Projects from Real REWA (Section 5.3) to Boolean REWA (Section 5.1)
//! using signed random projections (SimHash mechanism).

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// REWA Quantizer: Converts dense float vectors to binary signatures
pub struct RewaQuantizer {
    /// Random projection matrix (witness functions)
    /// Shape: (bit_depth, input_dim)
    projection_matrix: Array2<f32>,
    
    /// Number of bits in output signature
    bit_depth: usize,
    
    /// Input vector dimension
    input_dim: usize,
}

impl RewaQuantizer {
    /// Create a new REWA quantizer with random projection matrix
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of input vectors (e.g., 384 for MiniLM)
    /// * `bit_depth` - Number of bits in binary signature (e.g., 2048)
    /// * `seed` - Random seed for reproducibility
    pub fn new(input_dim: usize, bit_depth: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // Generate random Gaussian projection matrix (witness space)
        let projection_matrix = Array2::random_using(
            (bit_depth, input_dim),
            StandardNormal,
            &mut rng,
        );
        
        // TODO: Apply orthogonal rotation for balanced bit variance
        // (Section 6 mitigation for distribution skew)
        
        RewaQuantizer {
            projection_matrix,
            bit_depth,
            input_dim,
        }
    }
    
    /// Quantize a dense vector to binary signature
    ///
    /// # Arguments
    /// * `vector` - Input dense vector (should be L2-normalized)
    ///
    /// # Returns
    /// Binary signature as Vec<u64> (packed bits)
    pub fn quantize(&self, vector: &[f32]) -> Vec<u64> {
        assert_eq!(vector.len(), self.input_dim, "Vector dimension mismatch");
        
        let vec_array = Array1::from_vec(vector.to_vec());
        
        // Compute dot products: D = V · Codebook^T
        let projections = self.projection_matrix.dot(&vec_array);
        
        // Binarize: sign(D_i) -> {0, 1}
        let bits: Vec<bool> = projections.iter()
            .map(|&x| x > 0.0)
            .collect();
        
        // Pack into u64 array
        self.pack_bits(&bits)
    }
    
    /// Quantize multiple vectors in parallel
    pub fn quantize_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<u64>> {
        vectors.par_iter()
            .map(|v| self.quantize(v))
            .collect()
    }
    
    /// Pack boolean bits into u64 array for efficient storage
    fn pack_bits(&self, bits: &[bool]) -> Vec<u64> {
        let num_u64s = (self.bit_depth + 63) / 64;
        let mut packed = vec![0u64; num_u64s];
        
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                packed[word_idx] |= 1u64 << bit_idx;
            }
        }
        
        packed
    }
    
    /// Compute Hamming distance between two binary signatures
    ///
    /// Uses XOR + POPCNT (hardware instruction)
    pub fn hamming_distance(sig1: &[u64], sig2: &[u64]) -> u32 {
        assert_eq!(sig1.len(), sig2.len(), "Signature length mismatch");
        
        sig1.iter()
            .zip(sig2.iter())
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum()
    }
    
    /// Get projection matrix for inspection/debugging
    pub fn projection_matrix(&self) -> &Array2<f32> {
        &self.projection_matrix
    }
    
    /// Get bit depth
    pub fn bit_depth(&self) -> usize {
        self.bit_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_quantizer_creation() {
        let quantizer = RewaQuantizer::new(384, 2048, 42);
        assert_eq!(quantizer.bit_depth(), 2048);
        assert_eq!(quantizer.projection_matrix().shape(), &[2048, 384]);
    }
    
    #[test]
    fn test_quantize_vector() {
        let quantizer = RewaQuantizer::new(10, 64, 42);
        let vector = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0];
        
        let signature = quantizer.quantize(&vector);
        assert_eq!(signature.len(), 1); // 64 bits = 1 u64
    }
    
    #[test]
    fn test_hamming_distance() {
        let sig1 = vec![0b1010_1010u64];
        let sig2 = vec![0b1100_1100u64];
        
        let distance = RewaQuantizer::hamming_distance(&sig1, &sig2);
        assert_eq!(distance, 4); // 4 bits differ
    }
    
    #[test]
    fn test_simhash_property() {
        // Verify that similar vectors have low Hamming distance
        let quantizer = RewaQuantizer::new(10, 256, 42);
        
        let v1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        
        let sig1 = quantizer.quantize(&v1);
        let sig2 = quantizer.quantize(&v2);
        let sig3 = quantizer.quantize(&v3);
        
        let dist_similar = RewaQuantizer::hamming_distance(&sig1, &sig2);
        let dist_different = RewaQuantizer::hamming_distance(&sig1, &sig3);
        
        // Similar vectors should have lower Hamming distance
        assert!(dist_similar < dist_different);
    }
}
