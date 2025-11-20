//! Concept graph: nodes, edges, PMI weights, pruning, and metrics.

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
}
