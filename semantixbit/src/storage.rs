//! Binary Index: Flat storage for binary signatures with fast linear scan

use std::collections::HashMap;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

/// Binary signature storage and retrieval
#[derive(Serialize, Deserialize)]
pub struct BinaryIndex {
    /// Stored binary signatures (packed as u64 arrays)
    signatures: Vec<Vec<u64>>,
    
    /// Document IDs corresponding to signatures
    doc_ids: Vec<String>,
    
    /// Optional: Original vectors for zero-copy re-ranking
    original_vectors: Option<Vec<Vec<f32>>>,
    
    /// Bits per signature
    bit_depth: usize,
}

impl BinaryIndex {
    /// Create a new empty binary index
    pub fn new(bit_depth: usize) -> Self {
        BinaryIndex {
            signatures: Vec::new(),
            doc_ids: Vec::new(),
            original_vectors: None,
            bit_depth,
        }
    }
    
    /// Create index with original vector storage for re-ranking
    pub fn with_reranking(bit_depth: usize) -> Self {
        BinaryIndex {
            signatures: Vec::new(),
            doc_ids: Vec::new(),
            original_vectors: Some(Vec::new()),
            bit_depth,
        }
    }
    
    /// Add a document to the index
    ///
    /// # Arguments
    /// * `doc_id` - Unique document identifier
    /// * `signature` - Binary signature (packed u64 array)
    /// * `original_vector` - Optional original vector for re-ranking
    pub fn add(&mut self, doc_id: String, signature: Vec<u64>, original_vector: Option<Vec<f32>>) {
        self.signatures.push(signature);
        self.doc_ids.push(doc_id);
        
        if let Some(ref mut vectors) = self.original_vectors {
            if let Some(vec) = original_vector {
                vectors.push(vec);
            } else {
                panic!("Index configured for re-ranking but no vector provided");
            }
        }
    }
    
    /// Get number of documents in index
    pub fn len(&self) -> usize {
        self.signatures.len()
    }
    
    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }
    
    /// Get signature by index
    pub fn get_signature(&self, idx: usize) -> Option<&Vec<u64>> {
        self.signatures.get(idx)
    }
    
    /// Get document ID by index
    pub fn get_doc_id(&self, idx: usize) -> Option<&String> {
        self.doc_ids.get(idx)
    }
    
    /// Get original vector by index (if available)
    pub fn get_original_vector(&self, idx: usize) -> Option<&Vec<f32>> {
        self.original_vectors.as_ref()?.get(idx)
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let sig_bytes = self.signatures.len() * self.signatures[0].len() * 8;
        let id_bytes: usize = self.doc_ids.iter().map(|s| s.len()).sum();
        let vec_bytes = if let Some(ref vecs) = self.original_vectors {
            vecs.iter().map(|v| v.len() * 4).sum()
        } else {
            0
        };
        
        sig_bytes + id_bytes + vec_bytes
    }
    
    /// Save index to file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }
    
    /// Load index from file
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let index = bincode::deserialize(&data)?;
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_index_creation() {
        let index = BinaryIndex::new(2048);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }
    
    #[test]
    fn test_add_document() {
        let mut index = BinaryIndex::new(64);
        let signature = vec![0b1010_1010u64];
        
        index.add("doc1".to_string(), signature.clone(), None);
        
        assert_eq!(index.len(), 1);
        assert_eq!(index.get_doc_id(0), Some(&"doc1".to_string()));
        assert_eq!(index.get_signature(0), Some(&signature));
    }
    
    #[test]
    fn test_with_reranking() {
        let mut index = BinaryIndex::with_reranking(64);
        let signature = vec![0b1010_1010u64];
        let vector = vec![0.1, 0.2, 0.3];
        
        index.add("doc1".to_string(), signature, Some(vector.clone()));
        
        assert_eq!(index.get_original_vector(0), Some(&vector));
    }
}
