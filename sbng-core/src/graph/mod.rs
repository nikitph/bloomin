//! Concept graph: nodes, edges, PMI weights, pruning, and metrics.

use std::collections::HashMap;
use crate::types::ConceptId;

pub mod node;
pub mod edge;
pub mod builder;
pub mod pruning;
pub mod metrics;
/// Statistics accumulation for PMI calculation.
pub mod stats;

pub use node::ConceptNode;
pub use edge::ConceptEdge;
pub use builder::ConceptGraphBuilder;
pub use pruning::GraphPruner;
pub use metrics::GraphMetrics;
pub use stats::StatsAccumulator;

/// A high-level wrapper around the internal graph representation.
///
/// Internally, you can use `petgraph::Graph` or a custom adjacency structure.
#[derive(Debug)]
pub struct ConceptGraph {
    /// You can replace this with a more specialized structure later.
    inner: petgraph::graph::Graph<ConceptNode, ConceptEdge>,
    /// Precomputed degree map for O(1) lookups.
    pub degree_map: HashMap<ConceptId, u32>,
}

impl ConceptGraph {
    /// Access the underlying petgraph graph (for advanced operations).
    pub fn inner(&self) -> &petgraph::graph::Graph<ConceptNode, ConceptEdge> {
        &self.inner
    }
    
    /// Mutable access to the underlying petgraph graph.
    pub fn inner_mut(&mut self) -> &mut petgraph::graph::Graph<ConceptNode, ConceptEdge> {
        &mut self.inner
    }

    /// Recompute degrees for all concepts once, after build + pruning.
    pub fn recompute_degrees(&mut self) {
        let mut degs = HashMap::with_capacity(self.inner.node_count());

        for idx in self.inner.node_indices() {
            if let Some(node) = self.inner.node_weight(idx) {
                let degree = self.inner.neighbors(idx).count() as u32;
                degs.insert(node.id, degree);
            }
        }

        self.degree_map = degs;
    }

    /// O(1) degree lookup.
    pub fn degree(&self, cid: ConceptId) -> Option<u32> {
        self.degree_map.get(&cid).copied()
    }
}
