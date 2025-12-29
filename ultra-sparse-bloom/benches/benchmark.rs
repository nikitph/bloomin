use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ultra_sparse_bloom::{UltraSparseBloom, StandardBloomFilter};
use rand::{Rng, thread_rng};

fn benchmark_queries(c: &mut Criterion) {
    let n = 100_000; // Smaller scale for latency benchmarks to keep it fast
    let m = 500_000;
    let k = 5;
    let tau = 0.0001;

    let mut rng = thread_rng();
    let mut insert_data = Vec::with_capacity(n);
    for _ in 0..n {
        insert_data.push(rng.r#gen::<u64>());
    }

    let mut sbf = StandardBloomFilter::new(m as u64, k);
    let mut usb = UltraSparseBloom::new(m as u64, k, tau);

    for &x in &insert_data {
        sbf.insert(&x);
        usb.insert(x);
    }

    let query_val = insert_data[0];
    let miss_val = rng.r#gen::<u64>();

    let mut group = c.benchmark_group("Bloom Filter Queries");

    group.bench_function("Standard Bloom - Hit", |b| {
        b.iter(|| sbf.query(black_box(&query_val)))
    });

    group.bench_function("Standard Bloom - Miss", |b| {
        b.iter(|| sbf.query(black_box(&miss_val)))
    });

    group.bench_function("Ultra-Sparse Bloom - Hit", |b| {
        b.iter(|| usb.query(black_box(&query_val), 0.9, 0.1))
    });

    group.bench_function("Ultra-Sparse Bloom - Miss", |b| {
        b.iter(|| usb.query(black_box(&miss_val), 0.9, 0.1))
    });

    group.finish();
}

criterion_group!(benches, benchmark_queries);
criterion_main!(benches);
