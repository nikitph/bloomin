use pei::index::pei_index::PEIIndex;
use pei::storage::item_store::ItemStore;
use pei::evidence::dag::EvidenceDAG;
use pei::evidence::mock_ops::{FaissDistanceOp, MockCoarseOp};
use pei::evidence::operator::EvidenceOperator;
use pei::query::search::search;
use std::collections::HashMap;

fn main() {
    println!("Initializing PEI...");

    // 1. Create Mock Data
    // 1000 vectors of dim 4
    let mut vectors = Vec::new();
    for i in 0..1000 {
        let v = vec![
            (i as f32) * 0.1, 
            (i as f32) * 0.2, 
            0.0, 
            0.0
        ];
        vectors.push(v);
    }
    let store = ItemStore::new(vectors, vec![]); // Empty metadata

    // 2. Create Operators
    // Op 0: MockCoarse (Cheap)
    // Op 1: FaissDistance (Expensive)
    let op0 = MockCoarseOp { id: 0 };
    let op1 = FaissDistanceOp { id: 1 };
    
    let evidence_ops: Vec<Box<dyn EvidenceOperator>> = vec![
        Box::new(op0),
        Box::new(op1),
    ];

    // 3. Create DAG
    // Root: [0] (Coarse)
    // Edges: 0 -> [1] (Distance only available after Coarse)
    // This forces the "Filter then Refine" strategy
    let mut edges = HashMap::new();
    edges.insert(0, vec![1]);
    
    let dag = EvidenceDAG::new(vec![0], edges);

    // 4. Index
    let index = PEIIndex::new(evidence_ops, dag, store);

    // 5. Query
    let q_vec = vec![1.0, 2.0, 0.0, 0.0];
    let query = pei::query::Query::new(q_vec.clone());
    println!("Searching for query: {:?}", q_vec);

    let results = search(&index, &query, 10.0, 5);

    println!("Found {} results:", results.len());
    for (i, c) in results.iter().enumerate() {
        println!("{}. Item ID: {}, Belief: {:.4}, Uncertainty: {:.4}", 
                 i+1, c.item_id, c.belief, c.uncertainty);
    }
}
