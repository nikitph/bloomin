//! Hybrid Monoid: Compositional search combining semantic + keyword channels

use crate::quantizer::RewaQuantizer;
use twox_hash::XxHash32;
use std::hash::Hasher;

/// Hybrid encoder combining semantic and keyword channels
pub struct HybridEncoder {
    /// Semantic channel quantizer
    semantic_quantizer: RewaQuantizer,
    
    /// Number of bits for semantic channel
    semantic_bits: usize,
    
    /// Number of bits for keyword channel
    keyword_bits: usize,
    
    /// Total bit depth
    total_bits: usize,
}

impl HybridEncoder {
    /// Create a new hybrid encoder
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of input vectors
    /// * `semantic_bits` - Bits allocated to semantic channel
    /// * `keyword_bits` - Bits allocated to keyword channel
    /// * `seed` - Random seed
    pub fn new(input_dim: usize, semantic_bits: usize, keyword_bits: usize, seed: u64) -> Self {
        let semantic_quantizer = RewaQuantizer::new(input_dim, semantic_bits, seed);
        let total_bits = semantic_bits + keyword_bits;
        
        HybridEncoder {
            semantic_quantizer,
            semantic_bits,
            keyword_bits,
            total_bits,
        }
    }
    
    /// Encode vector and keywords into hybrid signature
    ///
    /// # Arguments
    /// * `vector` - Dense semantic vector
    /// * `keywords` - Optional keywords to hash into signature
    pub fn encode(&self, vector: &[f32], keywords: Option<&[String]>) -> Vec<u64> {
        // Get semantic signature
        let mut semantic_sig = self.semantic_quantizer.quantize(vector);
        
        // If no keywords, return semantic signature only
        let Some(kws) = keywords else {
            return semantic_sig;
        };
        
        // Hash keywords into keyword channel
        let keyword_sig = self.hash_keywords(kws);
        
        // Combine signatures
        self.combine_signatures(&semantic_sig, &keyword_sig)
    }
    
    /// Hash keywords into binary signature
    fn hash_keywords(&self, keywords: &[String]) -> Vec<u64> {
        let num_u64s = (self.keyword_bits + 63) / 64;
        let mut signature = vec![0u64; num_u64s];
        
        for keyword in keywords {
            // Hash keyword to bit positions
            let mut hasher = XxHash32::default();
            hasher.write(keyword.as_bytes());
            let hash = hasher.finish() as u32;
            let bit_pos = (hash as usize) % self.keyword_bits;
            
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            signature[word_idx] |= 1u64 << bit_idx;
        }
        
        signature
    }
    
    /// Combine semantic and keyword signatures
    fn combine_signatures(&self, semantic: &[u64], keyword: &[u64]) -> Vec<u64> {
        let total_u64s = (self.total_bits + 63) / 64;
        let mut combined = vec![0u64; total_u64s];
        
        // Copy semantic bits (first part)
        for (i, &val) in semantic.iter().enumerate() {
            combined[i] = val;
        }
        
        // Append keyword bits (second part)
        let semantic_u64s = semantic.len();
        for (i, &val) in keyword.iter().enumerate() {
            combined[semantic_u64s + i] = val;
        }
        
        combined
    }
    
    /// Get total bit depth
    pub fn total_bits(&self) -> usize {
        self.total_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hybrid_encoder() {
        let encoder = HybridEncoder::new(10, 128, 128, 42);
        assert_eq!(encoder.total_bits(), 256);
    }
    
    #[test]
    fn test_encode_semantic_only() {
        let encoder = HybridEncoder::new(10, 64, 64, 42);
        let vector = vec![0.1; 10];
        
        let signature = encoder.encode(&vector, None);
        assert!(signature.len() > 0);
    }
    
    #[test]
    fn test_encode_with_keywords() {
        let encoder = HybridEncoder::new(10, 64, 64, 42);
        let vector = vec![0.1; 10];
        let keywords = vec!["machine".to_string(), "learning".to_string()];
        
        let signature = encoder.encode(&vector, Some(&keywords));
        assert!(signature.len() > 0);
    }
    
    #[test]
    fn test_keyword_hashing() {
        let encoder = HybridEncoder::new(10, 64, 64, 42);
        let keywords = vec!["test".to_string()];
        
        let sig1 = encoder.hash_keywords(&keywords);
        let sig2 = encoder.hash_keywords(&keywords);
        
        // Same keywords should produce same signature
        assert_eq!(sig1, sig2);
    }
}
