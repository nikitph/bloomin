//! Graph-level and node-level metrics (degree, clustering, etc.).

use crate::graph::ConceptGraph;

/// Summary metrics for health-checking the graph.
#[derive(Debug, Default)]
pub struct GraphMetrics {
    /// Average degree of the graph.
    pub avg_degree: f32,
    /// Average clustering coefficient (placeholder).
    pub avg_clustering: f32,
    /// Total number of nodes.
    pub num_nodes: usize,
    /// Total number of edges.
    pub num_edges: usize,
}

impl GraphMetrics {
    /// Compute metrics for the given graph.
    pub fn compute(graph: &ConceptGraph) -> Self {
        let inner = graph.inner();
        let num_nodes = inner.node_count();
        let num_edges = inner.edge_count();
        
        let avg_degree = if num_nodes > 0 {
            (2 * num_edges) as f32 / num_nodes as f32
        } else {
            0.0
        };

        // TODO: implement clustering coefficient if needed, for now just basic stats
        Self {
            avg_degree,
            avg_clustering: 0.0, 
            num_nodes,
            num_edges,
        }
    }
}
