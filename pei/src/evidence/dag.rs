use std::collections::HashMap;
use bit_set::BitSet;

pub struct EvidenceDAG {
    pub roots: Vec<usize>,
    pub edges: HashMap<usize, Vec<usize>>,
}

impl EvidenceDAG {
    pub fn new(roots: Vec<usize>, edges: HashMap<usize, Vec<usize>>) -> Self {
        Self { roots, edges }
    }

    pub fn next_ops(&self, applied: &BitSet) -> Vec<usize> {
        let mut available = BitSet::new();
        
        // Add roots if they are not applied
        for &root in &self.roots {
            if !applied.contains(root) {
                available.insert(root);
            }
        }

        // Add children of all applied nodes
        for id in applied.iter() {
            if let Some(children) = self.edges.get(&id) {
                for &child in children {
                    if !applied.contains(child) {
                        available.insert(child);
                    }
                }
            }
        }

        available.into_iter().collect()
    }
}
