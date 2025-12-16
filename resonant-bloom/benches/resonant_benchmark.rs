//! Benchmarks for the Resonant Bloom Filter
//!
//! Compares performance of:
//! - Insertion: O(1) expected
//! - Time step: O(m) where m = number of buckets
//! - Query: O(1) expected
//! - Sequence query: O(1) expected

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use resonant_bloom::{EventCorrelator, ResonantBloom, ResonantConfig};
use std::f64::consts::PI;

fn benchmark_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut bloom = ResonantBloom::new(size);
            let mut rng = rand::thread_rng();
            let mut counter: u64 = 0;

            b.iter(|| {
                counter = counter.wrapping_add(rng.gen());
                bloom.insert(black_box(&counter));
            });
        });
    }
    group.finish();
}

fn benchmark_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_step");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut bloom = ResonantBloom::new(size);

            // Pre-populate
            for i in 0..100 {
                bloom.insert(&i);
            }

            b.iter(|| {
                bloom.step();
            });
        });
    }
    group.finish();
}

fn benchmark_step_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_step_n");

    let bloom_size = 4096;
    for n in [1, 10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            let mut bloom = ResonantBloom::new(bloom_size);

            for i in 0..100 {
                bloom.insert(&i);
            }

            b.iter(|| {
                bloom.step_n(black_box(n));
            });
        });
    }
    group.finish();
}

fn benchmark_query_amplitude(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_amplitude");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut bloom = ResonantBloom::new(size);

            // Populate
            for i in 0..1000 {
                bloom.insert(&i);
                if i % 10 == 0 {
                    bloom.step();
                }
            }

            let mut rng = rand::thread_rng();

            b.iter(|| {
                let key: u64 = rng.gen_range(0..1000);
                black_box(bloom.get_amplitude(black_box(&key)));
            });
        });
    }
    group.finish();
}

fn benchmark_sequence_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequence_query");

    for size in [256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut bloom = ResonantBloom::with_config(ResonantConfig {
                size,
                decay: 0.995,
                omega_base: 2.0 * PI / 128.0,
                seed: 42,
            });

            // Create a realistic scenario
            bloom.insert(&"event_a");
            bloom.step_n(50);
            bloom.insert(&"event_b");
            bloom.step_n(20);

            b.iter(|| {
                black_box(bloom.query_sequence(
                    black_box(&"event_a"),
                    black_box(&"event_b"),
                    black_box(50),
                ));
            });
        });
    }
    group.finish();
}

fn benchmark_event_correlator(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_correlator");

    group.bench_function("process_event", |b| {
        let mut correlator = EventCorrelator::new(4096, 200, 0.3);

        let events = ["login", "logout", "access", "error", "warning"];
        let mut rng = rand::thread_rng();
        let mut idx = 0;

        b.iter(|| {
            idx = (idx + 1) % events.len();
            correlator.process_event(black_box(&events[idx]));
        });
    });

    group.bench_function("check_correlation", |b| {
        let mut correlator = EventCorrelator::new(4096, 200, 0.3);

        // Setup scenario
        correlator.process_event(&"login_failed");
        correlator.advance(50);
        correlator.process_event(&"admin_access");
        correlator.advance(10);

        b.iter(|| {
            black_box(correlator.check_correlation(
                black_box(&"login_failed"),
                black_box(&"admin_access"),
                black_box(50),
            ));
        });
    });

    group.finish();
}

fn benchmark_high_throughput_stream(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_throughput");

    group.throughput(Throughput::Elements(10000));
    group.bench_function("10k_events_with_queries", |b| {
        b.iter(|| {
            let mut correlator = EventCorrelator::new(8192, 500, 0.3);

            // Simulate high-throughput stream
            for i in 0..10000u64 {
                correlator.process_event(&i);

                // Periodic queries (every 100 events)
                if i > 100 && i % 100 == 0 {
                    black_box(correlator.check_correlation(&(i - 100), &i, 100));
                }
            }
        });
    });

    group.finish();
}

fn benchmark_pattern_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_detection");

    let pattern: Vec<(&str, u64)> = vec![
        ("step1", 0),
        ("step2", 10),
        ("step3", 25),
        ("step4", 30),
        ("step5", 50),
    ];

    group.bench_function("detect_5_step_pattern", |b| {
        let mut bloom = ResonantBloom::with_config(ResonantConfig {
            size: 4096,
            decay: 0.99,
            omega_base: 2.0 * PI / 128.0,
            seed: 12345,
        });

        // Insert the pattern
        for (event, time) in &pattern {
            bloom.insert(event);
            bloom.step_n(*time);
        }

        b.iter(|| {
            black_box(bloom.detect_pattern(black_box(&pattern), 0.5));
        });
    });

    group.finish();
}

fn benchmark_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    // Measure insertion performance as filter fills up
    for load_factor in [0.1, 0.5, 1.0, 2.0, 5.0].iter() {
        let size = 4096;
        let num_insertions = (size as f64 * load_factor) as u64;

        group.bench_with_input(
            BenchmarkId::new("insert_at_load", format!("{:.1}x", load_factor)),
            &num_insertions,
            |b, &num_insertions| {
                let mut bloom = ResonantBloom::new(size);

                // Pre-load
                for i in 0..num_insertions {
                    bloom.insert(&i);
                    if i % 100 == 0 {
                        bloom.step();
                    }
                }

                let mut counter = num_insertions;

                b.iter(|| {
                    counter += 1;
                    bloom.insert(black_box(&counter));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_insertion,
    benchmark_step,
    benchmark_step_n,
    benchmark_query_amplitude,
    benchmark_sequence_query,
    benchmark_event_correlator,
    benchmark_high_throughput_stream,
    benchmark_pattern_detection,
    benchmark_memory_scaling,
);

criterion_main!(benches);
