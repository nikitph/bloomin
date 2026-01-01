use pei::index::pei_index::PEIIndex;
use pei::storage::item_store::ItemStore;
use pei::evidence::dag::EvidenceDAG;
use pei::evidence::mock_ops::{FaissDistanceOp, MockDimensionOp};
use pei::evidence::operator::EvidenceOperator;
use pei::query::search::search;
use pei::Query;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    println!("Generating Recall vs Budget Pareto Frontier...");

    // Setup High-Dim Data
    let size = 2000;
    let dim = 512;
    let mut vectors = Vec::new();
    for i in 0..size {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            let val = ((i + d) % 100) as f32 / 100.0;
            v.push(val);
        }
        vectors.push(v);
    }
    let store = ItemStore::new(vectors, vec![]); // empty metadata

    // Setup PEI Dual
    let op_dim0 = MockDimensionOp { id: 0, dim_index: 0 };
    let op_dim1 = MockDimensionOp { id: 1, dim_index: 1 };
    let op_dist = FaissDistanceOp { id: 2 };
    
    let ops: Vec<Box<dyn EvidenceOperator>> = vec![
        Box::new(op_dim0),
        Box::new(op_dim1),
        Box::new(op_dist),
    ];
    let mut edges = HashMap::new();
    edges.insert(0, vec![2]);
    edges.insert(1, vec![2]);
    let dag = EvidenceDAG::new(vec![0, 1], edges);
    let index = PEIIndex::new(ops, dag, ItemStore::new(store.vectors.clone(), vec![]));

    // Query
    let mut q_vec = Vec::with_capacity(dim);
    for d in 0..dim {
        q_vec.push(((50 + d) % 100) as f32 / 100.0);
    }
    let query = Query::new(q_vec);

    // Ground Truth (Standard Search)
    println!("Computing Ground Truth...");
    let k = 10;
    let mut cands = Vec::with_capacity(size);
    for i in 0..size {
        let d = store.distance(&query, i);
        cands.push((i, d));
    }
    cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ground_truth: Vec<usize> = cands.iter().take(k).map(|(id, _)| *id).collect();

    // PEI Budget Sweep
    let budgets = vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0];
    
    println!("| Budget | Time (µs) | Recall@10 |");
    println!("|--------|-----------|-----------|");

    for &budget in &budgets {
        let start = Instant::now();
        // Run loop 100 times for stability
        let mut results = Vec::new();
        for _ in 0..100 {
            results = search(&index, &query, budget, k);
        }
        let duration = start.elapsed();
        let avg_time_us = duration.as_micros() as f64 / 100.0;

        // Calculate Recall
        let mut hit = 0;
        for c in &results {
            if ground_truth.contains(&c.item_id) {
                hit += 1;
            }
        }
        let recall = hit as f64 / k as f64;

        println!("| {:.1} | {:.2} | {:.4} |", budget, avg_time_us, recall);
    }
}
