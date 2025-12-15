//! Merkle Tree Integrity Proofs
//!
//! This module provides cryptographic integrity guarantees for the H-Tree:
//! - Proof that a vector exists in the database
//! - Proof that the database has not been tampered with
//! - O(log N) proof size
//!
//! ## Integration with Witness Fields
//!
//! The Merkle hash incorporates both:
//! - The spectral Bloom filter (witness encoding)
//! - The actual content (children or vectors)
//!
//! This ensures that any modification to the witness field
//! is cryptographically detectable.

use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

use crate::node::NodeId;

/// A Merkle proof for a vector's existence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleProof {
    /// The path from leaf to root
    pub path: Vec<MerklePathEntry>,

    /// The vector ID being proven
    pub vector_id: u64,

    /// The root hash (commitment)
    pub root_hash: [u8; 32],
}

/// Entry in a Merkle proof path
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerklePathEntry {
    /// Node ID at this level
    pub node_id: NodeId,

    /// Hash of this node
    pub hash: [u8; 32],

    /// Sibling hashes (for verification)
    pub sibling_hashes: Vec<[u8; 32]>,

    /// Index of the child we descended through
    pub child_index: usize,
}

impl MerkleProof {
    /// Create a new proof
    pub fn new(vector_id: u64, root_hash: [u8; 32]) -> Self {
        Self {
            path: Vec::new(),
            vector_id,
            root_hash,
        }
    }

    /// Add an entry to the path
    pub fn add_entry(&mut self, entry: MerklePathEntry) {
        self.path.push(entry);
    }

    /// Verify the proof
    pub fn verify(&self) -> bool {
        if self.path.is_empty() {
            return false;
        }

        // Start from leaf, work up to root
        let mut current_hash = self.path[0].hash;

        for i in 1..self.path.len() {
            let entry = &self.path[i];

            // Combine with sibling hashes
            let mut hasher = Sha256::new();

            // Hash in order
            for (j, sibling_hash) in entry.sibling_hashes.iter().enumerate() {
                if j == entry.child_index {
                    hasher.update(&current_hash);
                }
                hasher.update(sibling_hash);
            }

            // Handle case where child is last
            if entry.child_index >= entry.sibling_hashes.len() {
                hasher.update(&current_hash);
            }

            let result = hasher.finalize();
            current_hash.copy_from_slice(&result);
        }

        // Final hash should match root
        current_hash == self.root_hash
    }

    /// Size of the proof in bytes
    pub fn size(&self) -> usize {
        let entry_size = 32 + 8 + 8; // hash + node_id + child_index
        let sibling_size: usize = self.path.iter()
            .map(|e| e.sibling_hashes.len() * 32)
            .sum();

        8 + 32 + self.path.len() * entry_size + sibling_size
    }
}

/// Merkle tree builder for batch operations
pub struct MerkleTreeBuilder {
    /// Leaf hashes
    leaves: Vec<[u8; 32]>,
}

impl MerkleTreeBuilder {
    pub fn new() -> Self {
        Self { leaves: Vec::new() }
    }

    /// Add a leaf hash
    pub fn add_leaf(&mut self, hash: [u8; 32]) {
        self.leaves.push(hash);
    }

    /// Compute the root hash
    pub fn compute_root(&self) -> [u8; 32] {
        if self.leaves.is_empty() {
            return [0u8; 32];
        }

        if self.leaves.len() == 1 {
            return self.leaves[0];
        }

        let mut level = self.leaves.clone();

        while level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in level.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    // Duplicate last hash for odd count
                    hasher.update(&chunk[0]);
                }

                let result = hasher.finalize();
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&result);
                next_level.push(hash);
            }

            level = next_level;
        }

        level[0]
    }
}

/// Hash combiner utility
pub fn combine_hashes(hashes: &[[u8; 32]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for hash in hashes {
        hasher.update(hash);
    }
    let result = hasher.finalize();
    let mut combined = [0u8; 32];
    combined.copy_from_slice(&result);
    combined
}

/// Hash arbitrary data
pub fn hash_data(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_builder() {
        let mut builder = MerkleTreeBuilder::new();

        builder.add_leaf(hash_data(b"leaf1"));
        builder.add_leaf(hash_data(b"leaf2"));
        builder.add_leaf(hash_data(b"leaf3"));
        builder.add_leaf(hash_data(b"leaf4"));

        let root = builder.compute_root();

        // Root should be deterministic
        let mut builder2 = MerkleTreeBuilder::new();
        builder2.add_leaf(hash_data(b"leaf1"));
        builder2.add_leaf(hash_data(b"leaf2"));
        builder2.add_leaf(hash_data(b"leaf3"));
        builder2.add_leaf(hash_data(b"leaf4"));

        assert_eq!(root, builder2.compute_root());
    }

    #[test]
    fn test_different_data_different_root() {
        let mut builder1 = MerkleTreeBuilder::new();
        builder1.add_leaf(hash_data(b"leaf1"));
        builder1.add_leaf(hash_data(b"leaf2"));

        let mut builder2 = MerkleTreeBuilder::new();
        builder2.add_leaf(hash_data(b"leaf1"));
        builder2.add_leaf(hash_data(b"leaf3")); // Different

        assert_ne!(builder1.compute_root(), builder2.compute_root());
    }
}
