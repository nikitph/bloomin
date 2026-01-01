use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pei::index::pei_index::PEIIndex;
use pei::storage::item_store::ItemStore;
use pei::evidence::dag::EvidenceDAG;
use pei::evidence::mock_ops::{FaissDistanceOp, MockCoarseOp};
use pei::evidence::operator::EvidenceOperator;
use pei::query::search::search;
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use pei::Query;

fn try_query_new(v: Vec<f32>) -> Query {
    Query::new(v)
}

// Helper for standard search baseline
#[derive(PartialEq)]
struct DistCand {
    id: usize,

    dist: f32,
}

impl Eq for DistCand {}

impl PartialOrd for DistCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // partial_cmp for f32 can return None, but we wrap.
        // We want MIN-heap for distance? No, max-heap usually, then keep size K.
        // OR: Max-heap of K items.
        // Simpler: Push all to heap (ordered by dist descending for pop min? No).
        // Let's just collect all distances and sort. That's "Brute Force".
        // Or standard Heap scan:
        // PriorityQueue usually pops MAX.
        // If we want smallest distance, we want to pop smallest? No, we pop the worst to discard?
        // Let's implement full scan + sort for simplicity of baseline (simulates standard flat index).
        // Or better: Max-Heap of size k.
        self.dist.partial_cmp(&other.dist).map(|o| o.reverse()) // Reverse for min-heap logic if needed?
    }
}

impl Ord for DistCand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

fn standard_search(store: &ItemStore, query: &Query, k: usize) {
    let mut cands = Vec::with_capacity(store.len());
    for i in 0..store.len() {
        let d = store.distance(query, i);
        cands.push((i, d));
    }
    cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let _ = &cands[0..k.min(cands.len())];
}


fn criteria_benchmark(c: &mut Criterion) {
    // High Dimensional Setup (512 dims)
    let size = 1000;
    let dim = 512;
    let mut vectors = Vec::new();
    // Create random-ish vectors
    // Using simple deterministic generator
    for i in 0..size {
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            let val = ((i + d) % 100) as f32 / 100.0;
            v.push(val);
        }
        vectors.push(v);
    }
    let store = ItemStore::new(vectors, vec![]);
    
    // Ops
    // Op0: Check Dim 0 (Cheap, Cost 0.05)
    // Op1: Check Dim 1 (Cheap, Cost 0.05)
    // Op2: Full Distance (Expensive, Cost 1.0)
    
    let op0 = pei::evidence::mock_ops::MockDimensionOp { id: 0, dim_index: 0 };
    let op1 = pei::evidence::mock_ops::MockDimensionOp { id: 1, dim_index: 1 };
    let op_dist = FaissDistanceOp { id: 2 };
    
    let evidence_ops: Vec<Box<dyn EvidenceOperator>> = vec![
        Box::new(op0),
        Box::new(op1),
        Box::new(op_dist),
    ];
    
    // DAG: 0, 1 are roots. 2 depends on them?
    // Let's say we can filter by 0 or 1.
    // Real PEI allows selecting any available.
    // Roots: [0, 1]. Edges: {0 -> [2], 1 -> [2]}.
    // This means we MUST do 0 or 1 before 2.
    // Actually, simple DAG: 0 -> 2.
    // This forces checking Dim 0 first.
    let mut edges = HashMap::new();
    edges.insert(0, vec![2]); 
    
    let dag = EvidenceDAG::new(vec![0], edges);
    
    let index = PEIIndex::new(evidence_ops, dag, store);
    
    let query = Query::new(vec![1.0, 2.0, 0.0, 0.0]);
    
    let mut group = c.benchmark_group("search_comparison");
    group.sample_size(50); // High dim is slower

    group.bench_function("pei_high_dim", |b| {
        b.iter(|| {
            search(
                black_box(&index), 
                black_box(&query), 
                black_box(100.0), 
                black_box(5)
            )
        })
    });

    group.bench_function("standard_high_dim", |b| {
        b.iter(|| {
            standard_search(black_box(&index.store), black_box(&query), black_box(5))
        })
    });

    group.finish();
}

criterion_group!(benches, criteria_benchmark);
criterion_main!(benches);
