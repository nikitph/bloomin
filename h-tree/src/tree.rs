//! H-Tree: The Holographic B-Tree
//!
//! This is the main data structure implementing thermodynamic vector search.
//!
//! ## Algorithm Overview
//!
//! ### Query (Thermodynamic Descent with LSH)
//! 1. Start at root
//! 2. Check vacuum state (O(1) rejection if no matches possible)
//! 3. Score all children using LSH-based spectral heat
//! 4. Descend to top-k hottest children (wide beam search)
//! 5. At leaf, perform linear search
//! 6. Parallel search across multiple branches
//!
//! ### Insert (B-Tree Style)
//! 1. Find a leaf with space (or create one via split)
//! 2. Insert vector into leaf
//! 3. Split if full, propagate splits up
//! 4. Update summaries along the path
//!
//! ## Complexity
//! - Query: O(log N) expected, O(1) vacuum rejection
//! - Insert: O(log N)
//! - Space: O(N log N) for summaries

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use rayon::prelude::*;

use crate::node::{HNode, NodeId, NodeContent, ChildEntry};
use crate::spectral::SpectralBloomFilter;
use crate::vector::Vector;
use crate::merkle::{MerkleProof, MerklePathEntry};

/// Configuration for the H-Tree
#[derive(Clone, Debug)]
pub struct HTreeConfig {
    /// Size of spectral Bloom filter
    pub filter_size: usize,

    /// Number of hash functions
    pub num_hashes: usize,

    /// Maximum fanout for internal nodes
    pub max_fanout: usize,

    /// Maximum vectors per leaf
    pub max_leaf_vectors: usize,

    /// Number of children to explore per query (beam width)
    pub beam_width: usize,

    /// Threshold for vacuum detection
    pub vacuum_threshold: f32,

    /// Lambda for negative evidence weighting
    pub negative_lambda: f32,
}

impl Default for HTreeConfig {
    fn default() -> Self {
        Self {
            filter_size: 256,      // Smaller filter with LSH
            num_hashes: 32,        // More hash bands for LSH
            max_fanout: 64,
            max_leaf_vectors: 64,
            beam_width: 16,        // Wider beam for better recall
            vacuum_threshold: 0.001,
            negative_lambda: 0.5,
        }
    }
}

impl HTreeConfig {
    /// High-recall configuration (slower but more accurate)
    pub fn high_recall() -> Self {
        Self {
            filter_size: 512,
            num_hashes: 64,
            max_fanout: 64,
            max_leaf_vectors: 64,
            beam_width: 48,        // Very wide beam for high recall
            vacuum_threshold: 0.0001,
            negative_lambda: 0.3,
        }
    }

    /// Fast configuration (lower recall but faster)
    pub fn fast() -> Self {
        Self {
            filter_size: 128,
            num_hashes: 16,
            max_fanout: 64,
            max_leaf_vectors: 64,
            beam_width: 8,
            vacuum_threshold: 0.01,
            negative_lambda: 0.5,
        }
    }
}

/// Query result with vector ID and similarity score
#[derive(Clone, Debug)]
pub struct QueryResult {
    pub vector_id: u64,
    pub similarity: f32,
}

impl Eq for QueryResult {}

impl PartialEq for QueryResult {
    fn eq(&self, other: &Self) -> bool {
        self.vector_id == other.vector_id
    }
}

impl Ord for QueryResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other.similarity.partial_cmp(&self.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for QueryResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Query statistics for benchmarking
#[derive(Clone, Debug, Default)]
pub struct QueryStats {
    pub nodes_visited: usize,
    pub vacuum_pruned: usize,
    pub leaves_searched: usize,
    pub vectors_compared: usize,
    pub root_vacuum: bool,
}

/// The Holographic B-Tree
pub struct HTree {
    pub config: HTreeConfig,
    nodes: HashMap<NodeId, HNode>,
    root_id: Option<NodeId>,
    next_id: NodeId,
    vector_count: u64,
    height: u32,
}

impl HTree {
    pub fn new(config: HTreeConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            root_id: None,
            next_id: 1,
            vector_count: 0,
            height: 0,
        }
    }

    pub fn default() -> Self {
        Self::new(HTreeConfig::default())
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Insert a vector into the tree using proper B-tree insertion
    pub fn insert(&mut self, vector: Vector) {
        // Handle empty tree
        if self.root_id.is_none() {
            let id = self.alloc_id();
            let mut root = HNode::new_leaf(id, self.config.filter_size, self.config.num_hashes);
            root.add_vector(vector);
            self.nodes.insert(id, root);
            self.root_id = Some(id);
            self.vector_count = 1;
            self.height = 1;
            return;
        }

        // Insert and handle potential splits
        let root_id = self.root_id.unwrap();
        let split_result = self.insert_recursive(root_id, vector);

        self.vector_count += 1;

        // If root split, create new root
        if let Some((new_node_id, new_summary)) = split_result {
            let old_root_id = self.root_id.unwrap();
            let old_summary = self.nodes.get(&old_root_id).unwrap().summary.clone();

            let new_root_id = self.alloc_id();
            let mut new_root = HNode::new_internal(
                new_root_id,
                self.height,
                self.config.filter_size,
                self.config.num_hashes
            );

            let old_count = self.count_vectors_in_subtree(old_root_id);
            let new_count = self.count_vectors_in_subtree(new_node_id);

            // Compute centroids for better routing
            let old_centroid = self.nodes.get(&old_root_id).and_then(|n| n.compute_centroid());
            let new_centroid = self.nodes.get(&new_node_id).and_then(|n| n.compute_centroid());

            new_root.add_child_with_centroid(old_root_id, old_summary, old_centroid, old_count);
            new_root.add_child_with_centroid(new_node_id, new_summary, new_centroid, new_count);

            // Update parent pointers
            if let Some(old_root) = self.nodes.get_mut(&old_root_id) {
                old_root.parent = Some(new_root_id);
            }
            if let Some(new_node) = self.nodes.get_mut(&new_node_id) {
                new_node.parent = Some(new_root_id);
            }

            self.nodes.insert(new_root_id, new_root);
            self.root_id = Some(new_root_id);
            self.height += 1;
        }
    }

    /// Recursive insert - returns Some((new_node_id, summary, centroid)) if split occurred
    fn insert_recursive(&mut self, node_id: NodeId, vector: Vector) -> Option<(NodeId, SpectralBloomFilter)> {
        let is_leaf = self.nodes.get(&node_id).map(|n| n.is_leaf()).unwrap_or(false);

        if is_leaf {
            // Insert into leaf
            {
                let leaf = self.nodes.get_mut(&node_id).unwrap();
                leaf.add_vector(vector);
            }

            // Check if split needed
            let needs_split = self.nodes.get(&node_id).unwrap().len() > self.config.max_leaf_vectors;

            if needs_split {
                return self.split_leaf(node_id);
            }
            return None;
        }

        // Internal node: find best child and recurse
        let child_id = self.find_insert_child(node_id, &vector);
        let split_result = self.insert_recursive(child_id, vector);

        // Update summary and centroids
        self.update_node_summary(node_id);

        // If child split, add the new child to this node
        if let Some((new_child_id, new_summary)) = split_result {
            let new_count = self.count_vectors_in_subtree(new_child_id);
            let new_centroid = self.nodes.get(&new_child_id).and_then(|n| n.compute_centroid());

            {
                let node = self.nodes.get_mut(&node_id).unwrap();
                node.add_child_with_centroid(new_child_id, new_summary, new_centroid, new_count);
            }

            // Update parent pointer
            if let Some(new_child) = self.nodes.get_mut(&new_child_id) {
                new_child.parent = Some(node_id);
            }

            // Check if this node now needs splitting
            let needs_split = self.nodes.get(&node_id).unwrap().len() > self.config.max_fanout;
            if needs_split {
                return self.split_internal(node_id);
            }
        }

        None
    }

    /// Find the best child to insert into (similarity-based with balancing)
    fn find_insert_child(&self, node_id: NodeId, vector: &Vector) -> NodeId {
        let node = self.nodes.get(&node_id).unwrap();

        if let NodeContent::Internal { children } = &node.content {
            // Use centroid similarity to choose the best subtree
            let mut best_child = children[0].node_id;
            let mut best_score = f32::NEG_INFINITY;

            for child in children {
                let sim_score = if let Some(ref centroid) = child.centroid {
                    Self::cosine_similarity(&vector.data, centroid)
                } else {
                    // Fallback: prefer less full children for balance
                    1.0 - (child.count as f32 / self.config.max_leaf_vectors as f32).min(1.0)
                };

                // Combine similarity with load factor to prevent over-filling
                let load_factor = child.count as f32 / (self.config.max_leaf_vectors as f32 * 2.0);
                let score = sim_score * (1.0 - load_factor * 0.3);  // Slight penalty for full nodes

                if score > best_score {
                    best_score = score;
                    best_child = child.node_id;
                }
            }

            best_child
        } else {
            node_id
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

    /// Split a leaf node - returns the new node's (id, summary)
    fn split_leaf(&mut self, node_id: NodeId) -> Option<(NodeId, SpectralBloomFilter)> {
        let vectors: Vec<Vector> = {
            let node = self.nodes.get(&node_id)?;
            if let NodeContent::Leaf { vectors } = &node.content {
                vectors.clone()
            } else {
                return None;
            }
        };

        let mid = vectors.len() / 2;

        // Keep first half in original node
        let filter_size = self.config.filter_size;
        let num_hashes = self.config.num_hashes;

        {
            let node = self.nodes.get_mut(&node_id)?;
            node.content = NodeContent::Leaf { vectors: Vec::new() };
            node.summary = SpectralBloomFilter::new(filter_size, num_hashes);

            for v in vectors.iter().take(mid) {
                if let NodeContent::Leaf { vectors: vecs } = &mut node.content {
                    node.summary.encode(v);
                    vecs.push(v.clone());
                }
            }
            node.update_merkle_hash();
        }

        // Create new node with second half
        let new_id = self.alloc_id();
        let parent = self.nodes.get(&node_id).and_then(|n| n.parent);
        let level = self.nodes.get(&node_id).map(|n| n.level).unwrap_or(0);

        let mut new_node = HNode::new_leaf(new_id, filter_size, num_hashes);
        new_node.level = level;
        new_node.parent = parent;

        for v in vectors.iter().skip(mid) {
            new_node.add_vector(v.clone());
        }

        let summary = new_node.summary.clone();
        self.nodes.insert(new_id, new_node);

        Some((new_id, summary))
    }

    /// Split an internal node - returns the new node's (id, summary)
    fn split_internal(&mut self, node_id: NodeId) -> Option<(NodeId, SpectralBloomFilter)> {
        let children: Vec<ChildEntry> = {
            let node = self.nodes.get(&node_id)?;
            if let NodeContent::Internal { children } = &node.content {
                children.clone()
            } else {
                return None;
            }
        };

        let mid = children.len() / 2;

        let filter_size = self.config.filter_size;
        let num_hashes = self.config.num_hashes;
        let level = self.nodes.get(&node_id).map(|n| n.level).unwrap_or(0);
        let parent = self.nodes.get(&node_id).and_then(|n| n.parent);

        // Keep first half in original node
        {
            let node = self.nodes.get_mut(&node_id)?;
            node.content = NodeContent::Internal { children: Vec::new() };
            node.summary = SpectralBloomFilter::new(filter_size, num_hashes);

            for child in children.iter().take(mid) {
                node.summary.merge(&child.summary);
                if let NodeContent::Internal { children: ch } = &mut node.content {
                    ch.push(child.clone());
                }
            }
            node.update_merkle_hash();
        }

        // Create new node with second half
        let new_id = self.alloc_id();
        let mut new_node = HNode::new_internal(new_id, level, filter_size, num_hashes);
        new_node.parent = parent;

        for child in children.iter().skip(mid) {
            new_node.add_child(child.node_id, child.summary.clone(), child.count);

            // Update child's parent pointer
            if let Some(child_node) = self.nodes.get_mut(&child.node_id) {
                child_node.parent = Some(new_id);
            }
        }

        let summary = new_node.summary.clone();
        self.nodes.insert(new_id, new_node);

        Some((new_id, summary))
    }

    /// Count vectors in a subtree
    fn count_vectors_in_subtree(&self, node_id: NodeId) -> u64 {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return 0,
        };

        match &node.content {
            NodeContent::Leaf { vectors } => vectors.len() as u64,
            NodeContent::Internal { children } => {
                children.iter().map(|c| c.count).sum()
            }
        }
    }

    /// Update a node's summary and centroids from its children
    fn update_node_summary(&mut self, node_id: NodeId) {
        // Collect child data including centroids
        let child_data: Vec<(NodeId, SpectralBloomFilter, Option<Vec<f32>>, u64)> = {
            let node = match self.nodes.get(&node_id) {
                Some(n) => n,
                None => return,
            };

            if let NodeContent::Internal { children } = &node.content {
                children.iter().filter_map(|c| {
                    self.nodes.get(&c.node_id).map(|n| {
                        let count = match &n.content {
                            NodeContent::Leaf { vectors } => vectors.len() as u64,
                            NodeContent::Internal { children: gc } => gc.iter().map(|g| g.count).sum(),
                        };
                        let centroid = n.compute_centroid();
                        (c.node_id, n.summary.clone(), centroid, count)
                    })
                }).collect()
            } else {
                return;
            }
        };

        if let Some(node) = self.nodes.get_mut(&node_id) {
            let size = node.summary.size;
            let num_hashes = node.summary.num_hashes;

            node.summary = SpectralBloomFilter::new(size, num_hashes);
            for (_, summary, _, _) in &child_data {
                node.summary.merge(summary);
            }

            if let NodeContent::Internal { children } = &mut node.content {
                for child in children.iter_mut() {
                    if let Some((_, summary, centroid, count)) = child_data.iter().find(|(id, _, _, _)| *id == child.node_id) {
                        child.summary = summary.clone();
                        child.centroid = centroid.clone();
                        child.count = *count;
                    }
                }
            }

            node.update_merkle_hash();
        }
    }

    /// Query for k-nearest neighbors
    pub fn query(&self, query: &Vector, k: usize) -> Vec<QueryResult> {
        let (results, _) = self.query_with_stats(query, k);
        results
    }

    /// Query with detailed statistics
    pub fn query_with_stats(&self, query: &Vector, k: usize) -> (Vec<QueryResult>, QueryStats) {
        let mut stats = QueryStats::default();

        let root_id = match self.root_id {
            Some(id) => id,
            None => return (Vec::new(), stats),
        };

        // Check vacuum at root (O(1) rejection)
        let root = self.nodes.get(&root_id).unwrap();
        if root.is_vacuum(query) {
            stats.root_vacuum = true;
            return (Vec::new(), stats);
        }

        // Beam search using priority queue
        let mut frontier: BinaryHeap<(OrderedFloat, NodeId)> = BinaryHeap::new();
        frontier.push((OrderedFloat(root.heat(query)), root_id));

        let mut results: Vec<QueryResult> = Vec::new();
        let mut visited_leaves = 0usize;
        let max_leaves = self.config.beam_width * 4; // Limit total leaves to search

        while let Some((OrderedFloat(_heat), node_id)) = frontier.pop() {
            if visited_leaves >= max_leaves {
                break;
            }

            stats.nodes_visited += 1;

            let node = match self.nodes.get(&node_id) {
                Some(n) => n,
                None => continue,
            };

            if node.is_vacuum(query) {
                stats.vacuum_pruned += 1;
                continue;
            }

            if node.is_leaf() {
                stats.leaves_searched += 1;
                visited_leaves += 1;

                let leaf_results = node.linear_search(query, k * 2); // Get more candidates
                stats.vectors_compared += node.len();

                for (vid, sim) in leaf_results {
                    results.push(QueryResult {
                        vector_id: vid,
                        similarity: sim,
                    });
                }
            } else {
                // Score ALL children, not just top-k, then add best to frontier
                let scores = node.score_children(query);

                for (idx, heat) in scores.iter().take(self.config.beam_width) {
                    if *heat < self.config.vacuum_threshold {
                        stats.vacuum_pruned += 1;
                        continue;
                    }

                    if let NodeContent::Internal { children } = &node.content {
                        let child_id = children[*idx].node_id;
                        frontier.push((OrderedFloat(*heat), child_id));
                    }
                }
            }
        }

        // Sort and deduplicate results
        results.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal)
        });
        results.dedup_by(|a, b| a.vector_id == b.vector_id);
        results.truncate(k);

        (results, stats)
    }

    /// Parallel query using rayon - searches multiple branches concurrently
    pub fn query_parallel(&self, query: &Vector, k: usize) -> Vec<QueryResult> {
        let root_id = match self.root_id {
            Some(id) => id,
            None => return Vec::new(),
        };

        // Check vacuum at root
        let root = match self.nodes.get(&root_id) {
            Some(r) => r,
            None => return Vec::new(),
        };

        if root.is_vacuum(query) {
            return Vec::new();
        }

        // Get all leaf nodes to search
        let leaves_to_search = self.get_hot_leaves(query, self.config.beam_width * 4);

        // Search leaves in parallel
        let results: Vec<QueryResult> = leaves_to_search
            .par_iter()
            .flat_map(|&leaf_id| {
                if let Some(node) = self.nodes.get(&leaf_id) {
                    if node.is_leaf() {
                        let leaf_results = node.linear_search(query, k);
                        return leaf_results.into_iter()
                            .map(|(vid, sim)| QueryResult { vector_id: vid, similarity: sim })
                            .collect::<Vec<_>>();
                    }
                }
                Vec::new()
            })
            .collect();

        // Merge and sort results
        let mut final_results = results;
        final_results.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal)
        });
        final_results.dedup_by(|a, b| a.vector_id == b.vector_id);
        final_results.truncate(k);

        final_results
    }

    /// Get the hottest leaf nodes to search
    fn get_hot_leaves(&self, query: &Vector, max_leaves: usize) -> Vec<NodeId> {
        let root_id = match self.root_id {
            Some(id) => id,
            None => return Vec::new(),
        };

        let mut frontier: BinaryHeap<(OrderedFloat, NodeId)> = BinaryHeap::new();
        let mut leaves = Vec::new();

        if let Some(root) = self.nodes.get(&root_id) {
            frontier.push((OrderedFloat(root.heat(query)), root_id));
        }

        while let Some((_, node_id)) = frontier.pop() {
            if leaves.len() >= max_leaves {
                break;
            }

            let node = match self.nodes.get(&node_id) {
                Some(n) => n,
                None => continue,
            };

            if node.is_leaf() {
                leaves.push(node_id);
            } else {
                // Add all children scored by heat
                let scores = node.score_children(query);
                for (idx, heat) in scores.iter().take(self.config.beam_width) {
                    if let NodeContent::Internal { children } = &node.content {
                        let child_id = children[*idx].node_id;
                        frontier.push((OrderedFloat(*heat), child_id));
                    }
                }
            }
        }

        leaves
    }

    /// Check if a query would return vacuum (no results) - O(1)
    /// Uses both spectral filter and centroid-based detection
    pub fn is_vacuum(&self, query: &Vector) -> bool {
        match self.root_id {
            Some(root_id) => {
                let root = match self.nodes.get(&root_id) {
                    Some(r) => r,
                    None => return true,
                };

                // Check spectral vacuum first
                if root.is_vacuum(query) {
                    return true;
                }

                // Also check centroid-based vacuum
                // If query is very dissimilar to all child centroids, it's likely vacuum
                if let NodeContent::Internal { children } = &root.content {
                    let max_similarity = children.iter()
                        .filter_map(|c| c.centroid.as_ref())
                        .map(|centroid| Self::cosine_similarity(&query.data, centroid))
                        .fold(f32::NEG_INFINITY, f32::max);

                    // If best centroid similarity is below threshold, likely vacuum
                    // For high-dimensional random data, cosine similarity ~ 0
                    // Matches typically have similarity > 0.3
                    if max_similarity < 0.1 {
                        return true;
                    }
                }

                false
            }
            None => true,
        }
    }

    /// Get the heat at root for a query
    pub fn root_heat(&self, query: &Vector) -> f32 {
        match self.root_id {
            Some(root_id) => {
                self.nodes.get(&root_id)
                    .map(|root| root.heat(query))
                    .unwrap_or(0.0)
            }
            None => 0.0,
        }
    }

    /// Generate a Merkle proof for a vector's existence
    pub fn generate_proof(&self, vector_id: u64) -> Option<MerkleProof> {
        let root_id = self.root_id?;
        let root = self.nodes.get(&root_id)?;

        let mut proof = MerkleProof::new(vector_id, root.merkle_hash);

        if self.build_proof_path(root_id, vector_id, &mut proof) {
            Some(proof)
        } else {
            None
        }
    }

    fn build_proof_path(&self, node_id: NodeId, vector_id: u64, proof: &mut MerkleProof) -> bool {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return false,
        };

        match &node.content {
            NodeContent::Leaf { vectors } => {
                if vectors.iter().any(|v| v.id == vector_id) {
                    proof.add_entry(MerklePathEntry {
                        node_id,
                        hash: node.merkle_hash,
                        sibling_hashes: Vec::new(),
                        child_index: 0,
                    });
                    return true;
                }
                false
            }
            NodeContent::Internal { children } => {
                for (idx, child) in children.iter().enumerate() {
                    if self.build_proof_path(child.node_id, vector_id, proof) {
                        let sibling_hashes: Vec<[u8; 32]> = children.iter()
                            .enumerate()
                            .filter(|(i, _)| *i != idx)
                            .filter_map(|(_, c)| self.nodes.get(&c.node_id).map(|n| n.merkle_hash))
                            .collect();

                        proof.add_entry(MerklePathEntry {
                            node_id,
                            hash: node.merkle_hash,
                            sibling_hashes,
                            child_index: idx,
                        });
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Get tree statistics
    pub fn stats(&self) -> TreeStats {
        TreeStats {
            vector_count: self.vector_count,
            node_count: self.nodes.len(),
            height: self.height,
            root_id: self.root_id,
            memory_bytes: self.memory_size(),
        }
    }

    pub fn memory_size(&self) -> usize {
        self.nodes.values().map(|n| n.memory_size()).sum()
    }

    pub fn verify_integrity(&self) -> bool {
        self.nodes.values().all(|node| node.verify_merkle_hash())
    }

    pub fn all_vectors(&self) -> Vec<&Vector> {
        let mut vectors = Vec::new();
        for node in self.nodes.values() {
            if let NodeContent::Leaf { vectors: vs } = &node.content {
                vectors.extend(vs.iter());
            }
        }
        vectors
    }
}

#[derive(Clone, Copy, Debug)]
struct OrderedFloat(f32);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Debug)]
pub struct TreeStats {
    pub vector_count: u64,
    pub node_count: usize,
    pub height: u32,
    pub root_id: Option<NodeId>,
    pub memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query() {
        let mut tree = HTree::default();

        for i in 0..100 {
            let v = Vector::random(i, 128);
            tree.insert(v);
        }

        assert_eq!(tree.vector_count, 100);

        let query = Vector::random(1000, 128);
        let results = tree.query(&query, 10);

        assert!(results.len() <= 10);
    }

    #[test]
    fn test_tree_balance() {
        let config = HTreeConfig {
            max_leaf_vectors: 8,
            max_fanout: 8,
            ..Default::default()
        };
        let mut tree = HTree::new(config);

        for i in 0..1000 {
            let v = Vector::random(i, 32);
            tree.insert(v);
        }

        let stats = tree.stats();

        // Height should be logarithmic: log_8(1000) ≈ 3.3
        assert!(stats.height <= 6, "Tree height {} is too large", stats.height);
    }

    #[test]
    fn test_recall() {
        let mut tree = HTree::default();
        let dim = 64;

        let mut vectors: Vec<Vector> = Vec::new();
        for i in 0..200 {
            let v = Vector::random(i, dim);
            vectors.push(v.clone());
            tree.insert(v);
        }

        let query = &vectors[50];
        let results = tree.query(query, 10);

        let found = results.iter().any(|r| r.vector_id == 50);
        assert!(found, "Query vector should be found in results");
    }
}
