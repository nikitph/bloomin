//! H-Tree Node Structure
//!
//! Each node in the H-Tree contains:
//! - Spectral summaries of all vectors in its subtree (the "hologram")
//! - Child pointers (for internal nodes) or vectors (for leaf nodes)
//! - Merkle hash for integrity verification
//! - Metadata for lazy propagation
//!
//! ## Page Layout (4KB aligned)
//!
//! The node is designed to fit in a single disk page:
//! - Spectral summary: ~2KB
//! - Child pointers/vectors: ~1.5KB
//! - Merkle hash + metadata: ~0.5KB

use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

use crate::spectral::SpectralBloomFilter;
use crate::vector::Vector;

/// Maximum children per internal node (fanout)
pub const MAX_FANOUT: usize = 64;

/// Maximum vectors per leaf node
pub const MAX_LEAF_VECTORS: usize = 64;

/// Lazy propagation threshold
pub const PROPAGATION_THRESHOLD: f64 = 0.1;

/// Batch size for lazy propagation
pub const PROPAGATION_BATCH_SIZE: u32 = 16;

/// Unique identifier for nodes
pub type NodeId = u64;

/// H-Tree Node - the fundamental unit of the holographic index
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HNode {
    /// Unique identifier
    pub id: NodeId,

    /// Spectral Bloom filter summarizing all vectors in subtree
    /// This is the "hologram" - encodes the witness field
    pub summary: SpectralBloomFilter,

    /// Level in tree (0 = leaf)
    pub level: u32,

    /// Node contents
    pub content: NodeContent,

    /// Merkle hash for integrity
    pub merkle_hash: [u8; 32],

    /// Dirty counter for lazy propagation
    pub dirty_counter: u32,

    /// Pending delta (accumulated changes not yet propagated)
    pub pending_delta: Option<SpectralBloomFilter>,

    /// Parent node ID (None for root)
    pub parent: Option<NodeId>,
}

/// Node content - either children (internal) or vectors (leaf)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeContent {
    /// Internal node: contains child node IDs with their spectral summaries
    Internal {
        children: Vec<ChildEntry>,
    },

    /// Leaf node: contains actual vectors
    Leaf {
        vectors: Vec<Vector>,
    },
}

/// Entry for a child node in an internal node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChildEntry {
    /// Child node ID
    pub node_id: NodeId,

    /// Cached spectral summary of child
    /// This enables scoring without loading child from disk
    pub summary: SpectralBloomFilter,

    /// Centroid of vectors in child subtree (for similarity-based routing)
    pub centroid: Option<Vec<f32>>,

    /// Vector count in child subtree
    pub count: u64,
}

/// Propagation signal returned after insert
#[derive(Clone, Debug)]
pub struct PropagationSignal {
    /// Delta to positive evidence
    pub positive_delta: SpectralBloomFilter,

    /// Delta to negative evidence
    pub negative_delta: Option<SpectralBloomFilter>,

    /// Whether full propagation is needed
    pub needs_propagation: bool,
}

impl PropagationSignal {
    pub fn new(filter: SpectralBloomFilter) -> Self {
        Self {
            positive_delta: filter,
            negative_delta: None,
            needs_propagation: true,
        }
    }

    pub fn empty() -> Self {
        Self {
            positive_delta: SpectralBloomFilter::default(),
            negative_delta: None,
            needs_propagation: false,
        }
    }
}

impl HNode {
    /// Create a new leaf node
    pub fn new_leaf(id: NodeId, filter_size: usize, num_hashes: usize) -> Self {
        Self {
            id,
            summary: SpectralBloomFilter::new(filter_size, num_hashes),
            level: 0,
            content: NodeContent::Leaf { vectors: Vec::new() },
            merkle_hash: [0u8; 32],
            dirty_counter: 0,
            pending_delta: None,
            parent: None,
        }
    }

    /// Create a new internal node
    pub fn new_internal(id: NodeId, level: u32, filter_size: usize, num_hashes: usize) -> Self {
        Self {
            id,
            summary: SpectralBloomFilter::new(filter_size, num_hashes),
            level,
            content: NodeContent::Internal { children: Vec::new() },
            merkle_hash: [0u8; 32],
            dirty_counter: 0,
            pending_delta: None,
            parent: None,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        matches!(self.content, NodeContent::Leaf { .. })
    }

    /// Check if this is an internal node
    pub fn is_internal(&self) -> bool {
        matches!(self.content, NodeContent::Internal { .. })
    }

    /// Get number of children (internal) or vectors (leaf)
    pub fn len(&self) -> usize {
        match &self.content {
            NodeContent::Internal { children } => children.len(),
            NodeContent::Leaf { vectors } => vectors.len(),
        }
    }

    /// Check if node needs splitting
    pub fn needs_split(&self) -> bool {
        match &self.content {
            NodeContent::Internal { children } => children.len() >= MAX_FANOUT,
            NodeContent::Leaf { vectors } => vectors.len() >= MAX_LEAF_VECTORS,
        }
    }

    /// Compute heat score for a query
    ///
    /// This is the thermodynamic potential that guides search
    pub fn heat(&self, query: &Vector) -> f32 {
        self.summary.heat(query)
    }

    /// Check if node is in vacuum state for query
    ///
    /// This enables O(1) rejection: if vacuum, guaranteed no matches
    pub fn is_vacuum(&self, query: &Vector) -> bool {
        self.summary.is_vacuum(query)
    }

    /// Score all children and return indices sorted by similarity (descending)
    /// Uses centroid similarity if available, falls back to heat
    pub fn score_children(&self, query: &Vector) -> Vec<(usize, f32)> {
        match &self.content {
            NodeContent::Internal { children } => {
                let mut scores: Vec<(usize, f32)> = children
                    .iter()
                    .enumerate()
                    .map(|(idx, child)| {
                        // Prefer centroid-based similarity when available
                        let score = if let Some(ref centroid) = child.centroid {
                            Self::cosine_similarity(&query.data, centroid)
                        } else {
                            // Fallback to spectral heat
                            child.summary.heat(query)
                        };
                        (idx, score)
                    })
                    .collect();

                // Sort by score descending
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scores
            }
            NodeContent::Leaf { .. } => Vec::new(),
        }
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 1e-10 && norm_b > 1e-10 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Compute centroid of vectors in this node (for leaf nodes)
    pub fn compute_centroid(&self) -> Option<Vec<f32>> {
        match &self.content {
            NodeContent::Leaf { vectors } => {
                if vectors.is_empty() {
                    return None;
                }
                let dim = vectors[0].dim();
                let mut centroid = vec![0.0f32; dim];

                for v in vectors {
                    for (i, val) in v.data.iter().enumerate() {
                        centroid[i] += val;
                    }
                }

                let n = vectors.len() as f32;
                for val in &mut centroid {
                    *val /= n;
                }

                // Normalize
                let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for val in &mut centroid {
                        *val /= norm;
                    }
                }

                Some(centroid)
            }
            NodeContent::Internal { children } => {
                // Combine child centroids weighted by count
                let mut total_count = 0u64;
                let mut dim = 0usize;

                // Find dimension
                for child in children {
                    if let Some(ref c) = child.centroid {
                        dim = c.len();
                        break;
                    }
                }

                if dim == 0 {
                    return None;
                }

                let mut centroid = vec![0.0f32; dim];

                for child in children {
                    if let Some(ref c) = child.centroid {
                        let weight = child.count as f32;
                        total_count += child.count;
                        for (i, val) in c.iter().enumerate() {
                            centroid[i] += val * weight;
                        }
                    }
                }

                if total_count == 0 {
                    return None;
                }

                let n = total_count as f32;
                for val in &mut centroid {
                    *val /= n;
                }

                // Normalize
                let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for val in &mut centroid {
                        *val /= norm;
                    }
                }

                Some(centroid)
            }
        }
    }

    /// Get top-k hottest children
    pub fn top_k_children(&self, query: &Vector, k: usize) -> Vec<(usize, f32)> {
        let mut scores = self.score_children(query);
        scores.truncate(k);
        scores
    }

    /// Add a vector to a leaf node
    pub fn add_vector(&mut self, vector: Vector) -> PropagationSignal {
        // Encode in summary first
        self.summary.encode(&vector);

        // Create delta for propagation
        let mut delta = SpectralBloomFilter::new(self.summary.size, self.summary.num_hashes);
        delta.encode(&vector);

        // Add to vector list
        match &mut self.content {
            NodeContent::Leaf { vectors } => {
                vectors.push(vector);
            }
            NodeContent::Internal { .. } => {
                panic!("Cannot add vector to internal node");
            }
        }

        // Update dirty counter
        self.dirty_counter += 1;

        // Update Merkle hash
        self.update_merkle_hash();

        // Return propagation signal
        PropagationSignal {
            positive_delta: delta,
            negative_delta: None,
            needs_propagation: self.should_propagate(),
        }
    }

    /// Add a child to an internal node
    pub fn add_child(&mut self, child_id: NodeId, child_summary: SpectralBloomFilter, count: u64) {
        self.add_child_with_centroid(child_id, child_summary, None, count);
    }

    /// Add a child with centroid to an internal node
    pub fn add_child_with_centroid(
        &mut self,
        child_id: NodeId,
        child_summary: SpectralBloomFilter,
        centroid: Option<Vec<f32>>,
        count: u64
    ) {
        match &mut self.content {
            NodeContent::Internal { children } => {
                // Merge child summary into this node's summary
                self.summary.merge(&child_summary);

                // Add child entry
                children.push(ChildEntry {
                    node_id: child_id,
                    summary: child_summary,
                    centroid,
                    count,
                });

                // Update Merkle hash
                self.update_merkle_hash();
            }
            NodeContent::Leaf { .. } => {
                panic!("Cannot add child to leaf node");
            }
        }
    }

    /// Apply propagation signal from child
    pub fn apply_propagation(&mut self, signal: &PropagationSignal) {
        if !signal.needs_propagation {
            return;
        }

        // Merge delta into summary
        self.summary.merge(&signal.positive_delta);

        // Merge negative delta if present
        if let Some(ref neg_delta) = signal.negative_delta {
            for i in 0..self.summary.size {
                self.summary.negative[i] = self.summary.negative[i].saturating_add(neg_delta.positive[i]);
            }
        }

        self.dirty_counter += 1;
        self.update_merkle_hash();
    }

    /// Check if changes should be propagated to parent
    fn should_propagate(&self) -> bool {
        // Propagate if:
        // 1. Dirty counter exceeds batch size
        // 2. Or significant change in summary magnitude
        self.dirty_counter >= PROPAGATION_BATCH_SIZE
    }

    /// Reset dirty counter (after propagation)
    pub fn reset_dirty(&mut self) {
        self.dirty_counter = 0;
        self.pending_delta = None;
    }

    /// Linear search within leaf node
    pub fn linear_search(&self, query: &Vector, k: usize) -> Vec<(u64, f32)> {
        match &self.content {
            NodeContent::Leaf { vectors } => {
                let mut results: Vec<(u64, f32)> = vectors
                    .iter()
                    .map(|v| (v.id, query.cosine_similarity(v)))
                    .collect();

                // Sort by similarity descending
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(k);
                results
            }
            NodeContent::Internal { .. } => Vec::new(),
        }
    }

    /// Compute Merkle hash from contents
    pub fn update_merkle_hash(&mut self) {
        let mut hasher = Sha256::new();

        // Hash the summary
        hasher.update(&self.summary.to_bytes());

        // Hash the content
        match &self.content {
            NodeContent::Internal { children } => {
                for child in children {
                    hasher.update(&child.node_id.to_le_bytes());
                    hasher.update(&child.summary.to_bytes());
                }
            }
            NodeContent::Leaf { vectors } => {
                for v in vectors {
                    hasher.update(&v.id.to_le_bytes());
                    for f in &v.data {
                        hasher.update(&f.to_le_bytes());
                    }
                }
            }
        }

        let result = hasher.finalize();
        self.merkle_hash.copy_from_slice(&result);
    }

    /// Verify Merkle hash
    pub fn verify_merkle_hash(&self) -> bool {
        let mut hasher = Sha256::new();

        hasher.update(&self.summary.to_bytes());

        match &self.content {
            NodeContent::Internal { children } => {
                for child in children {
                    hasher.update(&child.node_id.to_le_bytes());
                    hasher.update(&child.summary.to_bytes());
                }
            }
            NodeContent::Leaf { vectors } => {
                for v in vectors {
                    hasher.update(&v.id.to_le_bytes());
                    for f in &v.data {
                        hasher.update(&f.to_le_bytes());
                    }
                }
            }
        }

        let result = hasher.finalize();
        result.as_slice() == self.merkle_hash
    }

    /// Estimate memory footprint
    pub fn memory_size(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let summary_size = self.summary.memory_size();

        let content_size = match &self.content {
            NodeContent::Internal { children } => {
                children.len() * (std::mem::size_of::<ChildEntry>() +
                    children.first().map(|c| c.summary.memory_size()).unwrap_or(0))
            }
            NodeContent::Leaf { vectors } => {
                vectors.iter().map(|v| std::mem::size_of::<Vector>() + v.data.len() * 4).sum()
            }
        };

        base + summary_size + content_size
    }

    /// Get child entries (for internal nodes)
    pub fn children(&self) -> Option<&Vec<ChildEntry>> {
        match &self.content {
            NodeContent::Internal { children } => Some(children),
            NodeContent::Leaf { .. } => None,
        }
    }

    /// Get vectors (for leaf nodes)
    pub fn vectors(&self) -> Option<&Vec<Vector>> {
        match &self.content {
            NodeContent::Leaf { vectors } => Some(vectors),
            NodeContent::Internal { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaf_node_operations() {
        let mut node = HNode::new_leaf(1, 256, 4);

        let v1 = Vector::new(1, vec![1.0, 0.0, 0.0, 0.0]);
        let v2 = Vector::new(2, vec![0.0, 1.0, 0.0, 0.0]);

        node.add_vector(v1.clone());
        node.add_vector(v2.clone());

        assert_eq!(node.len(), 2);
        assert!(node.is_leaf());

        // Heat should be non-zero for encoded vectors
        assert!(node.heat(&v1) > 0.0);
    }

    #[test]
    fn test_merkle_integrity() {
        let mut node = HNode::new_leaf(1, 256, 4);

        let v1 = Vector::new(1, vec![1.0, 2.0, 3.0, 4.0]);
        node.add_vector(v1);

        assert!(node.verify_merkle_hash());

        // Tamper with summary
        node.summary.positive[0] = 255;

        assert!(!node.verify_merkle_hash());
    }

    #[test]
    fn test_vacuum_detection() {
        let mut node = HNode::new_leaf(1, 256, 4);

        let v1 = Vector::new(1, vec![1.0, 0.0, 0.0, 0.0]);
        let v2 = Vector::new(2, vec![100.0, 200.0, 300.0, 400.0]);

        // Initially vacuum for all queries
        assert!(node.is_vacuum(&v1));
        assert!(node.is_vacuum(&v2));

        // Encode v1
        node.add_vector(v1.clone());

        // No longer vacuum for v1
        assert!(!node.is_vacuum(&v1));
    }
}
