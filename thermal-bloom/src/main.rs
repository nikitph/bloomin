use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// CORE DATA STRUCTURES
// ============================================================================

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_sq(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

// ============================================================================
// DISCRETE BLOOM FILTER (BASELINE)
// ============================================================================

pub struct DiscreteBloom {
    grid: Vec<Vec<u8>>,
    items: HashMap<(usize, usize), usize>,
    grid_size: usize,
    range_min: f64,
    range_max: f64,
}

impl DiscreteBloom {
    pub fn new(grid_size: usize, range_min: f64, range_max: f64) -> Self {
        Self {
            grid: vec![vec![0u8; grid_size]; grid_size],
            items: HashMap::new(),
            grid_size,
            range_min,
            range_max,
        }
    }

    fn hash(&self, point: &Point2D) -> (usize, usize) {
        let scale = (self.grid_size as f64 - 1.0) / (self.range_max - self.range_min);
        let x = ((point.x - self.range_min) * scale).clamp(0.0, self.grid_size as f64 - 1.0) as usize;
        let y = ((point.y - self.range_min) * scale).clamp(0.0, self.grid_size as f64 - 1.0) as usize;
        (x, y)
    }

    pub fn insert(&mut self, point: &Point2D, item_id: usize) {
        let (x, y) = self.hash(point);
        self.grid[x][y] = 1;
        self.items.insert((x, y), item_id);
    }

    pub fn query(&self, point: &Point2D) -> Option<usize> {
        let (x, y) = self.hash(point);
        if self.grid[x][y] > 0 {
            self.items.get(&(x, y)).copied()
        } else {
            None
        }
    }

    pub fn get_grid(&self) -> &Vec<Vec<u8>> {
        &self.grid
    }
}

// ============================================================================
// THERMAL BLOOM V2 - WITH MULTI-ITEM STORAGE AND CANDIDATE RANKING
// ============================================================================

pub struct ThermalBloomV2 {
    grid: Vec<Vec<f64>>,
    items: HashMap<(usize, usize), Vec<usize>>,  // Store ALL items per cell
    points: Vec<Point2D>,  // Keep original points for distance computation
    grid_size: usize,
    sigma: f64,
    range_min: f64,
    range_max: f64,
}

impl ThermalBloomV2 {
    pub fn new(grid_size: usize, sigma: f64, range_min: f64, range_max: f64) -> Self {
        Self {
            grid: vec![vec![0.0; grid_size]; grid_size],
            items: HashMap::new(),
            points: Vec::new(),
            grid_size,
            sigma,
            range_min,
            range_max,
        }
    }

    fn hash(&self, point: &Point2D) -> (usize, usize) {
        let scale = (self.grid_size as f64 - 1.0) / (self.range_max - self.range_min);
        let x = ((point.x - self.range_min) * scale).clamp(0.0, self.grid_size as f64 - 1.0) as usize;
        let y = ((point.y - self.range_min) * scale).clamp(0.0, self.grid_size as f64 - 1.0) as usize;
        (x, y)
    }

    pub fn insert(&mut self, point: &Point2D, item_id: usize) {
        let (x, y) = self.hash(point);
        self.grid[x][y] = 1.0;
        self.items.entry((x, y)).or_default().push(item_id);

        // Ensure points vector is large enough
        if self.points.len() <= item_id {
            self.points.resize(item_id + 1, Point2D::new(0.0, 0.0));
        }
        self.points[item_id] = *point;
    }

    pub fn finalize(&mut self) {
        let kernel_radius = (self.sigma * 3.0).ceil() as i32;
        let kernel_size = (kernel_radius * 2 + 1) as usize;
        let mut kernel = vec![vec![0.0; kernel_size]; kernel_size];
        let mut kernel_sum = 0.0;

        for i in 0..kernel_size {
            for j in 0..kernel_size {
                let dx = i as f64 - kernel_radius as f64;
                let dy = j as f64 - kernel_radius as f64;
                let value = (-((dx * dx + dy * dy) / (2.0 * self.sigma * self.sigma))).exp();
                kernel[i][j] = value;
                kernel_sum += value;
            }
        }

        for row in kernel.iter_mut() {
            for val in row.iter_mut() {
                *val /= kernel_sum;
            }
        }

        let mut new_grid = vec![vec![0.0; self.grid_size]; self.grid_size];

        for x in 0..self.grid_size {
            for y in 0..self.grid_size {
                let mut sum = 0.0;
                for ki in 0..kernel_size {
                    for kj in 0..kernel_size {
                        let gx = x as i32 + ki as i32 - kernel_radius;
                        let gy = y as i32 + kj as i32 - kernel_radius;

                        if gx >= 0 && gx < self.grid_size as i32
                           && gy >= 0 && gy < self.grid_size as i32 {
                            sum += self.grid[gx as usize][gy as usize] * kernel[ki][kj];
                        }
                    }
                }
                new_grid[x][y] = sum;
            }
        }

        self.grid = new_grid;
    }

    /// Get all candidates from a cell
    fn get_candidates(&self, x: usize, y: usize) -> Vec<usize> {
        self.items.get(&(x, y)).cloned().unwrap_or_default()
    }

    /// Query with gradient ascent and candidate ranking
    pub fn query(&self, query_point: &Point2D, max_steps: usize, search_radius: i32) -> (Option<usize>, usize, usize) {
        let (mut x, mut y) = self.hash(query_point);
        let mut visited_cells: Vec<(usize, usize)> = vec![(x, y)];

        // Gradient ascent
        for step in 0..max_steps {
            if x == 0 || x >= self.grid_size - 1 || y == 0 || y >= self.grid_size - 1 {
                break;
            }

            let dx = (self.grid[x + 1][y] - self.grid[x - 1][y]) / 2.0;
            let dy = (self.grid[x][y + 1] - self.grid[x][y - 1]) / 2.0;

            let grad_mag = (dx * dx + dy * dy).sqrt();
            if grad_mag < 1e-8 {
                break;
            }

            let new_x = if dx > 0.001 {
                x + 1
            } else if dx < -0.001 {
                x.saturating_sub(1)
            } else {
                x
            };

            let new_y = if dy > 0.001 {
                y + 1
            } else if dy < -0.001 {
                y.saturating_sub(1)
            } else {
                y
            };

            if new_x == x && new_y == y {
                break;
            }

            x = new_x;
            y = new_y;
            visited_cells.push((x, y));
        }

        // Collect ALL candidates from neighborhood around termination point
        let mut all_candidates: Vec<usize> = Vec::new();

        for dx in -search_radius..=search_radius {
            for dy in -search_radius..=search_radius {
                let nx = (x as i32 + dx).clamp(0, self.grid_size as i32 - 1) as usize;
                let ny = (y as i32 + dy).clamp(0, self.grid_size as i32 - 1) as usize;
                all_candidates.extend(self.get_candidates(nx, ny));
            }
        }

        // Also collect from visited cells along the path
        for (vx, vy) in &visited_cells {
            all_candidates.extend(self.get_candidates(*vx, *vy));
        }

        let num_candidates = all_candidates.len();

        // Rank candidates by actual distance
        if all_candidates.is_empty() {
            return (None, visited_cells.len(), 0);
        }

        let best = all_candidates
            .into_iter()
            .min_by(|&a, &b| {
                let dist_a = query_point.distance_sq(&self.points[a]);
                let dist_b = query_point.distance_sq(&self.points[b]);
                dist_a.partial_cmp(&dist_b).unwrap()
            });

        (best, visited_cells.len(), num_candidates)
    }

    pub fn get_grid(&self) -> &Vec<Vec<f64>> {
        &self.grid
    }
}

// ============================================================================
// DATA GENERATION
// ============================================================================

pub fn make_blobs(
    n_samples: usize,
    n_centers: usize,
    cluster_std: f64,
    range_min: f64,
    range_max: f64,
    seed: u64,
) -> (Vec<Point2D>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, cluster_std).unwrap();

    let range = range_max - range_min;
    let margin = range * 0.1;
    let centers: Vec<Point2D> = (0..n_centers)
        .map(|_| {
            Point2D::new(
                rng.gen_range((range_min + margin)..(range_max - margin)),
                rng.gen_range((range_min + margin)..(range_max - margin)),
            )
        })
        .collect();

    let mut points = Vec::with_capacity(n_samples);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let center_idx = i % n_centers;
        let center = &centers[center_idx];

        let x = (center.x + normal.sample(&mut rng)).clamp(range_min, range_max);
        let y = (center.y + normal.sample(&mut rng)).clamp(range_min, range_max);

        points.push(Point2D::new(x, y));
        labels.push(center_idx);
    }

    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let shuffled_points: Vec<Point2D> = indices.iter().map(|&i| points[i]).collect();
    let shuffled_labels: Vec<usize> = indices.iter().map(|&i| labels[i]).collect();

    (shuffled_points, shuffled_labels)
}

// ============================================================================
// K-NN GROUND TRUTH
// ============================================================================

pub fn compute_knn(index_points: &[Point2D], query_points: &[Point2D], k: usize) -> Vec<Vec<usize>> {
    query_points
        .par_iter()
        .map(|query| {
            let mut distances: Vec<(usize, f64)> = index_points
                .iter()
                .enumerate()
                .map(|(i, p)| (i, query.distance_sq(p)))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect()
}

// ============================================================================
// BENCHMARK METRICS
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub method: String,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub avg_steps: f64,
    pub avg_candidates: f64,
    pub queries_per_second: f64,
    pub total_queries: usize,
    pub sigma: Option<f64>,
    pub grid_size: Option<usize>,
    pub search_radius: Option<i32>,
}

pub fn compute_recall(
    results: &[Option<usize>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> f64 {
    let hits: usize = results
        .iter()
        .zip(ground_truth.iter())
        .filter(|(result, truth)| {
            if let Some(r) = result {
                truth.iter().take(k).any(|t| t == r)
            } else {
                false
            }
        })
        .count();

    hits as f64 / results.len() as f64
}

// ============================================================================
// VISUALIZATION DATA OUTPUT
// ============================================================================

#[derive(Serialize)]
pub struct VisualizationData {
    pub discrete_grid: Vec<Vec<u8>>,
    pub thermal_grid: Vec<Vec<f64>>,
    pub index_points: Vec<Point2D>,
    pub query_points: Vec<Point2D>,
    pub benchmark_results: Vec<BenchmarkResult>,
    pub parameter_sweep: Vec<BenchmarkResult>,
}

// ============================================================================
// BENCHMARKS
// ============================================================================

fn benchmark_thermal_v2(
    x_index: &[Point2D],
    x_query: &[Point2D],
    ground_truth: &[Vec<usize>],
    grid_size: usize,
    sigma: f64,
    search_radius: i32,
    range_min: f64,
    range_max: f64,
) -> BenchmarkResult {
    let mut thermal = ThermalBloomV2::new(grid_size, sigma, range_min, range_max);
    for (i, point) in x_index.iter().enumerate() {
        thermal.insert(point, i);
    }
    thermal.finalize();

    let start = Instant::now();
    let results: Vec<(Option<usize>, usize, usize)> = x_query
        .iter()
        .map(|q| thermal.query(q, 50, search_radius))
        .collect();
    let elapsed = start.elapsed().as_secs_f64();

    let items: Vec<Option<usize>> = results.iter().map(|(item, _, _)| *item).collect();
    let total_steps: usize = results.iter().map(|(_, steps, _)| steps).sum();
    let total_candidates: usize = results.iter().map(|(_, _, cands)| cands).sum();
    let avg_steps = total_steps as f64 / x_query.len() as f64;
    let avg_candidates = total_candidates as f64 / x_query.len() as f64;

    let recall_1 = compute_recall(&items, ground_truth, 1);
    let recall_5 = compute_recall(&items, ground_truth, 5);
    let qps = x_query.len() as f64 / elapsed;

    BenchmarkResult {
        method: format!("ThermalV2_g{}_s{}_r{}", grid_size, sigma, search_radius),
        recall_at_1: recall_1,
        recall_at_5: recall_5,
        avg_steps,
        avg_candidates,
        queries_per_second: qps,
        total_queries: x_query.len(),
        sigma: Some(sigma),
        grid_size: Some(grid_size),
        search_radius: Some(search_radius),
    }
}

fn run_comprehensive_benchmark(
    range_min: f64,
    range_max: f64,
) -> Vec<BenchmarkResult> {
    println!("\n{}", "=".repeat(70));
    println!("THERMAL BLOOM V2 - COMPREHENSIVE BENCHMARK");
    println!("{}", "=".repeat(70));

    // Generate data
    println!("\nGenerating synthetic clustered data...");
    let n_samples = 10000;
    let n_centers = 10;
    let cluster_std = 1.5;

    let (points, _) = make_blobs(n_samples, n_centers, cluster_std, range_min, range_max, 42);

    let split_idx = (n_samples as f64 * 0.8) as usize;
    let x_index = &points[..split_idx];
    let x_query = &points[split_idx..];

    println!("  Index: {} points, Query: {} points", x_index.len(), x_query.len());

    // Compute ground truth
    println!("Computing k-NN ground truth...");
    let ground_truth = compute_knn(x_index, x_query, 5);

    // Discrete baseline
    println!("\nBuilding Discrete Bloom baseline...");
    let mut discrete = DiscreteBloom::new(256, range_min, range_max);
    for (i, point) in x_index.iter().enumerate() {
        discrete.insert(point, i);
    }
    let start = Instant::now();
    let discrete_results: Vec<Option<usize>> = x_query.iter().map(|q| discrete.query(q)).collect();
    let discrete_time = start.elapsed().as_secs_f64();
    let discrete_recall_1 = compute_recall(&discrete_results, &ground_truth, 1);
    let discrete_recall_5 = compute_recall(&discrete_results, &ground_truth, 5);
    println!("  Discrete Bloom: Recall@1={:.1}%, Recall@5={:.1}%",
             discrete_recall_1 * 100.0, discrete_recall_5 * 100.0);

    let mut results = Vec::new();
    results.push(BenchmarkResult {
        method: "DiscreteBloom".to_string(),
        recall_at_1: discrete_recall_1,
        recall_at_5: discrete_recall_5,
        avg_steps: 0.0,
        avg_candidates: 1.0,
        queries_per_second: x_query.len() as f64 / discrete_time,
        total_queries: x_query.len(),
        sigma: None,
        grid_size: Some(256),
        search_radius: None,
    });

    // Parameter grid search for Thermal V2
    println!("\n{}", "-".repeat(70));
    println!("PARAMETER GRID SEARCH");
    println!("{}", "-".repeat(70));
    println!("{:>6} {:>6} {:>6} | {:>8} {:>8} {:>6} {:>6} {:>10}",
             "Grid", "Sigma", "Radius", "R@1", "R@5", "Steps", "Cands", "QPS");
    println!("{}", "-".repeat(70));

    let grid_sizes = [128, 256, 512];
    let sigmas = [0.5, 1.0, 2.0, 3.0];
    let search_radii = [2, 3, 5];

    for &grid_size in &grid_sizes {
        for &sigma in &sigmas {
            for &search_radius in &search_radii {
                let result = benchmark_thermal_v2(
                    x_index, x_query, &ground_truth,
                    grid_size, sigma, search_radius,
                    range_min, range_max,
                );

                println!("{:>6} {:>6.1} {:>6} | {:>7.1}% {:>7.1}% {:>6.1} {:>6.1} {:>10.0}",
                         grid_size, sigma, search_radius,
                         result.recall_at_1 * 100.0,
                         result.recall_at_5 * 100.0,
                         result.avg_steps,
                         result.avg_candidates,
                         result.queries_per_second);

                results.push(result);
            }
        }
    }

    results
}

fn run_scaling_test(range_min: f64, range_max: f64) {
    println!("\n{}", "=".repeat(70));
    println!("SCALING TEST");
    println!("{}", "=".repeat(70));

    let dataset_sizes = [5000, 10000, 20000, 50000, 100000];
    let grid_size = 512;
    let sigma = 1.0;
    let search_radius = 3;

    println!("{:>10} {:>10} {:>10} {:>8} {:>8} {:>12}",
             "N_index", "N_query", "Build(ms)", "R@1", "R@5", "QPS");
    println!("{}", "-".repeat(70));

    for &n_samples in &dataset_sizes {
        let (points, _) = make_blobs(n_samples, 10, 1.5, range_min, range_max, 42);
        let split_idx = (n_samples as f64 * 0.8) as usize;
        let x_index = &points[..split_idx];
        let x_query = &points[split_idx..];

        // Build index
        let build_start = Instant::now();
        let mut thermal = ThermalBloomV2::new(grid_size, sigma, range_min, range_max);
        for (i, point) in x_index.iter().enumerate() {
            thermal.insert(point, i);
        }
        thermal.finalize();
        let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

        // Compute ground truth
        let ground_truth = compute_knn(x_index, x_query, 5);

        // Query
        let start = Instant::now();
        let results: Vec<(Option<usize>, usize, usize)> = x_query
            .iter()
            .map(|q| thermal.query(q, 50, search_radius))
            .collect();
        let elapsed = start.elapsed().as_secs_f64();

        let items: Vec<Option<usize>> = results.iter().map(|(item, _, _)| *item).collect();
        let recall_1 = compute_recall(&items, &ground_truth, 1);
        let recall_5 = compute_recall(&items, &ground_truth, 5);
        let qps = x_query.len() as f64 / elapsed;

        println!("{:>10} {:>10} {:>10.1} {:>7.1}% {:>7.1}% {:>12.0}",
                 x_index.len(), x_query.len(), build_time,
                 recall_1 * 100.0, recall_5 * 100.0, qps);
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("THERMAL BLOOM FILTER V2 - PoC VALIDATION");
    println!("Multi-item storage with candidate ranking");
    println!("{}", "=".repeat(70));

    let range_min = -10.0;
    let range_max = 10.0;

    // Comprehensive parameter search
    let results = run_comprehensive_benchmark(range_min, range_max);

    // Scaling test
    run_scaling_test(range_min, range_max);

    // Find best configuration
    let best = results
        .iter()
        .filter(|r| r.method.starts_with("Thermal"))
        .max_by(|a, b| a.recall_at_1.partial_cmp(&b.recall_at_1).unwrap())
        .unwrap();

    let discrete = results.iter().find(|r| r.method == "DiscreteBloom").unwrap();

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("FINAL SUMMARY");
    println!("{}", "=".repeat(70));

    println!("\nBest Configuration: {}", best.method);
    println!("  Grid: {}x{}", best.grid_size.unwrap(), best.grid_size.unwrap());
    println!("  Sigma: {:.1}", best.sigma.unwrap());
    println!("  Search Radius: {}", best.search_radius.unwrap());
    println!("  Recall@1: {:.1}%", best.recall_at_1 * 100.0);
    println!("  Recall@5: {:.1}%", best.recall_at_5 * 100.0);
    println!("  Avg Steps: {:.1}", best.avg_steps);
    println!("  Avg Candidates: {:.1}", best.avg_candidates);
    println!("  QPS: {:.0}", best.queries_per_second);

    // Success criteria
    println!("\n{}", "-".repeat(40));
    println!("SUCCESS CRITERIA CHECK:");
    println!("{}", "-".repeat(40));

    let mvp_passed = best.recall_at_1 > 0.5 && best.avg_steps < 20.0;
    let strong_passed = best.recall_at_1 > 0.7;
    let holy_passed = best.recall_at_1 > 0.85;

    println!("  [{}] MVP: Recall@1 > 50%, Steps < 20", if mvp_passed { "PASS" } else { "FAIL" });
    println!("  [{}] Strong: Recall@1 > 70%", if strong_passed { "PASS" } else { "FAIL" });
    println!("  [{}] Holy Shit: Recall@1 > 85%", if holy_passed { "PASS" } else { "FAIL" });

    println!("\nThermal V2: {:.1}% Recall@1", best.recall_at_1 * 100.0);
    println!("Discrete Bloom: {:.1}% Recall@1", discrete.recall_at_1 * 100.0);
    println!("Improvement: {:.1}x",
             if discrete.recall_at_1 > 0.0 { best.recall_at_1 / discrete.recall_at_1 } else { f64::INFINITY });

    // Save results
    let json = serde_json::to_string_pretty(&results).unwrap();
    std::fs::write("thermal_bloom_v2_results.json", &json).unwrap();
    println!("\nSaved: thermal_bloom_v2_results.json");
}
