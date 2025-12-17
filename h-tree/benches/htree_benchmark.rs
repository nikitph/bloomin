//! Criterion benchmarks for H-Tree

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use h_tree::{HTree, Vector};
use h_tree::tree::HTreeConfig;

fn benchmark_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for size in [100, 1000, 5000].iter() {
        let vectors: Vec<Vector> = (0..*size)
            .map(|i| Vector::random(i as u64, 128))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut tree = HTree::default();
                for v in &vectors {
                    tree.insert(black_box(v.clone()));
                }
                tree
            });
        });
    }

    group.finish();
}

fn benchmark_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");

    // Pre-build tree
    let mut tree = HTree::default();
    let vectors: Vec<Vector> = (0..10000)
        .map(|i| Vector::random(i, 128))
        .collect();

    for v in &vectors {
        tree.insert(v.clone());
    }

    let query = Vector::random(99999, 128);

    for k in [1, 10, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(k), k, |b, k| {
            b.iter(|| {
                tree.query(black_box(&query), *k)
            });
        });
    }

    group.finish();
}

fn benchmark_vacuum_detection(c: &mut Criterion) {
    let mut tree = HTree::default();

    // Insert vectors in specific region
    for i in 0..5000 {
        let mut data = vec![0.0f32; 128];
        data[0] = i as f32;
        tree.insert(Vector::new(i, data));
    }

    // Query in distant region
    let query = Vector::new(99999, vec![1000.0; 128]);

    c.bench_function("vacuum_detection", |b| {
        b.iter(|| {
            tree.is_vacuum(black_box(&query))
        });
    });
}

criterion_group!(benches, benchmark_insert, benchmark_query, benchmark_vacuum_detection);
criterion_main!(benches);
