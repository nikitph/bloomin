use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pei::index::pei_index::PEIIndex;
use pei::storage::item_store::ItemStore;
use pei::evidence::dag::EvidenceDAG;
use pei::evidence::mock_ops::{FaissDistanceOp, MockDimensionOp};
use pei::evidence::operator::EvidenceOperator;
use pei::query::search::search;
use pei::Query;
use std::collections::HashMap;

// Re-implement standard search helper locally to avoid linking issues
fn standard_search(store: &ItemStore, query: &Query, k: usize) {
    let mut cands = Vec::with_capacity(store.len());
    for i in 0..store.len() {
        let d = store.distance(query, i);
        cands.push((i, d));
    }
    cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let _ = &cands[0..k.min(cands.len())];
}

fn multi_modal_benchmark(c: &mut Criterion) {
    // Large High-Dim Setup
    let size = 2000;
    let dim = 512;
    let mut vectors = Vec::new();
    for i in 0..size {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            // Deterministic pattern
            let val = ((i + d) % 100) as f32 / 100.0;
            v.push(val);
        }
        vectors.push(v);
    }
    let store = ItemStore::new(vectors, vec![]);

    // Setup 1: Single Cheap Op (Dim 0)
    let op_dim0 = MockDimensionOp { id: 0, dim_index: 0 };
    // Dist is ID 1
    let ops_single: Vec<Box<dyn EvidenceOperator>> = vec![
        Box::new(op_dim0),
        Box::new(FaissDistanceOp { id: 1 }),
    ];
    let mut edges_single = HashMap::new();
    edges_single.insert(0, vec![1]); 
    let dag_single = EvidenceDAG::new(vec![0], edges_single);
    let index_single = PEIIndex::new(ops_single, dag_single, ItemStore::new(store.vectors.clone(), vec![]));


    // Setup 2: Dual Cheap Ops (Dim 0 AND Dim 1)
    let op_dim0_dual = MockDimensionOp { id: 0, dim_index: 0 };
    let op_dim1_dual = MockDimensionOp { id: 1, dim_index: 1 };
    // Dist is ID 2
    let ops_dual: Vec<Box<dyn EvidenceOperator>> = vec![
        Box::new(op_dim0_dual),
        Box::new(op_dim1_dual),
        Box::new(FaissDistanceOp { id: 2 }),
    ];
    
    let mut edges_dual = HashMap::new();
    edges_dual.insert(0, vec![2]);
    edges_dual.insert(1, vec![2]);
    
    let dag_dual = EvidenceDAG::new(vec![0, 1], edges_dual);
    let index_dual = PEIIndex::new(ops_dual, dag_dual, ItemStore::new(store.vectors.clone(), vec![]));


    // Query
    let mut q_vec = Vec::with_capacity(dim);
    for d in 0..dim {
        q_vec.push(((50 + d) % 100) as f32 / 100.0);
    }
    let query = Query::new(q_vec);

    let mut group = c.benchmark_group("multi_modal_pruning");
    group.sample_size(40);

    // 1. Standard (Baseline)
    group.bench_function("1_standard_baseline", |b| {
        b.iter(|| {
            standard_search(black_box(&index_single.store), black_box(&query), black_box(5))
        })
    });

    // 2. PEI Single Filter
    group.bench_function("2_pei_single_filter", |b| {
        b.iter(|| {
            search(
                black_box(&index_single), 
                black_box(&query), 
                black_box(200.0), 
                black_box(5)
            )
        })
    });

    // 3. PEI Dual Filter (Expected Winner)
    group.bench_function("3_pei_dual_filter", |b| {
        b.iter(|| {
            search(
                black_box(&index_dual), 
                black_box(&query), 
                black_box(200.0), 
                black_box(5)
            )
        })
    });
}

criterion_group!(benches, multi_modal_benchmark);
criterion_main!(benches);
