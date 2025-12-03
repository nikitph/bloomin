//! UNIFIED INVESTIGATION SUITE: Probing the Rough Edges of Semantic Thermodynamics
//!
//! This program runs 5 targeted experiments to investigate:
//! 1. K-Dependence: Does C_M increase with K? (Explaining the mismatch)
//! 2. N-Scaling: Does χ_c scale logarithmically with N? (Core theory validation)
//! 3. Rho-Dependence: Is universality robust across signal strengths?
//! 4. L-Dependence: Does witness count matter?
//! 5. Precision: High-resolution transition analysis.

use ndarray::Array2;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

use plotters::prelude::*;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Clone)]
struct REWALab {
    n: usize,
    w_universe: usize,
    l: usize,
    items: Vec<HashSet<usize>>,
    cluster_labels: Vec<usize>,
    seed: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExperimentResult {
    k_values: Option<Vec<usize>>,
    n_values: Option<Vec<usize>>,
    rho_values: Option<Vec<f64>>,
    l_values: Option<Vec<usize>>,
    chi_c_values: Option<Vec<f64>>,
    chi_c: Option<f64>,
    chi_c_err: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AllResults {
    exp1: ExperimentResult,
    exp2: ExperimentResult,
    exp3: ExperimentResult,
    exp4: ExperimentResult,
    exp5: ExperimentResult,
}

// ============================================================================
// REWA LABORATORY IMPLEMENTATION
// ============================================================================

impl REWALab {
    fn new(n: usize, w_universe: usize, l: usize, seed: u64) -> Self {
        REWALab {
            n,
            w_universe,
            l,
            items: Vec::new(),
            cluster_labels: Vec::new(),
            seed,
        }
    }

    fn generate_clustered_data(&mut self, k_clusters: usize, rho: f64) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let items_per_cluster = self.n / k_clusters;
        let shared_count = (rho * self.l as f64) as usize;
        let unique_count = self.l - shared_count;

        // Ensure we have enough witnesses
        let required_witnesses = k_clusters * shared_count + self.n * unique_count;
        if required_witnesses > self.w_universe {
            self.w_universe = (required_witnesses as f64 * 1.2) as usize;
        }

        // Create shuffled witness pool
        let mut witness_pool: Vec<usize> = (0..self.w_universe).collect();
        witness_pool.shuffle(&mut rng);
        let mut witness_id = 0;

        self.items.clear();
        self.cluster_labels.clear();

        for cluster_idx in 0..k_clusters {
            // Cluster shared witnesses
            let cluster_shared: HashSet<usize> = witness_pool[witness_id..witness_id + shared_count]
                .iter()
                .copied()
                .collect();
            witness_id += shared_count;

            for _ in 0..items_per_cluster {
                let unique_witnesses: HashSet<usize> =
                    witness_pool[witness_id..witness_id + unique_count]
                        .iter()
                        .copied()
                        .collect();
                witness_id += unique_count;

                let mut item_witnesses = cluster_shared.clone();
                item_witnesses.extend(unique_witnesses);
                self.items.push(item_witnesses);
                self.cluster_labels.push(cluster_idx);
            }
        }
    }

    fn measure_gap(&self, n_samples: usize) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed + 1000);
        let mut same_cluster_overlaps = Vec::new();
        let mut diff_cluster_overlaps = Vec::new();

        for _ in 0..n_samples {
            let i = rng.gen_range(0..self.n);
            let mut j = rng.gen_range(0..self.n);
            while j == i {
                j = rng.gen_range(0..self.n);
            }

            let overlap =
                self.items[i].intersection(&self.items[j]).count() as f64 / self.l as f64;

            if self.cluster_labels[i] == self.cluster_labels[j] {
                same_cluster_overlaps.push(overlap);
            } else {
                diff_cluster_overlaps.push(overlap);
            }
        }

        let delta_same = if same_cluster_overlaps.is_empty() {
            0.0
        } else {
            same_cluster_overlaps.iter().sum::<f64>() / same_cluster_overlaps.len() as f64
        };
        let delta_diff = if diff_cluster_overlaps.is_empty() {
            0.0
        } else {
            diff_cluster_overlaps.iter().sum::<f64>() / diff_cluster_overlaps.len() as f64
        };

        delta_same - delta_diff
    }

    fn encode(&self, m: usize, k_hashes: usize, seed: u64) -> Array2<i16> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Create hash map: W_universe x K_hashes -> [0, m)
        let hash_map: Vec<Vec<usize>> = (0..self.w_universe)
            .map(|_| (0..k_hashes).map(|_| rng.gen_range(0..m)).collect())
            .collect();

        let mut encoded_matrix = Array2::<i16>::zeros((self.n, m));

        for (i, witnesses) in self.items.iter().enumerate() {
            for &w in witnesses {
                for &hash_idx in &hash_map[w] {
                    encoded_matrix[[i, hash_idx]] = 1;
                }
            }
        }

        encoded_matrix
    }

    fn evaluate_retrieval(&self, encoded_matrix: &Array2<i16>) -> f64 {
        let n = encoded_matrix.nrows();
        let mut correct = 0;

        // Compute similarity matrix via dot product
        for i in 0..n {
            let row_i = encoded_matrix.row(i);
            let mut max_sim = i32::MIN;
            let mut nearest_idx = i;

            for j in 0..n {
                if i == j {
                    continue;
                }
                let row_j = encoded_matrix.row(j);
                let sim: i32 = row_i
                    .iter()
                    .zip(row_j.iter())
                    .map(|(&a, &b)| (a as i32) * (b as i32))
                    .sum();

                if sim > max_sim {
                    max_sim = sim;
                    nearest_idx = j;
                }
            }

            if self.cluster_labels[i] == self.cluster_labels[nearest_idx] {
                correct += 1;
            }
        }

        correct as f64 / n as f64
    }
}

// ============================================================================
// SIGMOID FITTING
// ============================================================================

fn sigmoid(x: f64, x_c: f64, a: f64, y_min: f64, y_max: f64) -> f64 {
    y_min + (y_max - y_min) / (1.0 + (-a * (x - x_c)).exp())
}

/// Fit sigmoid to find critical point chi_c using grid search + refinement
fn fit_critical_point(m_values: &[usize], accuracies: &[f64], delta: f64) -> Option<(f64, f64)> {
    let x_scaled: Vec<f64> = m_values.iter().map(|&m| m as f64 * delta * delta).collect();

    if x_scaled.is_empty() || accuracies.is_empty() {
        return None;
    }

    let x_min = x_scaled.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Grid search for best parameters
    let mut best_loss = f64::INFINITY;
    let mut best_params = (x_scaled[x_scaled.len() / 2], 0.05, 0.0, 1.0);

    // Search grid
    for x_c in (0..50).map(|i| x_min + (x_max - x_min) * i as f64 / 50.0) {
        for a in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] {
            for y_min in [0.0, 0.05, 0.1] {
                for y_max in [0.85, 0.9, 0.95, 1.0] {
                    let loss: f64 = x_scaled
                        .iter()
                        .zip(accuracies.iter())
                        .map(|(&x, &y)| {
                            let pred = sigmoid(x, x_c, a, y_min, y_max);
                            (pred - y).powi(2)
                        })
                        .sum();

                    if loss < best_loss {
                        best_loss = loss;
                        best_params = (x_c, a, y_min, y_max);
                    }
                }
            }
        }
    }

    // Refinement around best params
    let (mut x_c, mut a, y_min, y_max) = best_params;
    for _ in 0..100 {
        let delta_xc = 0.1;
        let delta_a = 0.001;

        for dx in [-delta_xc, 0.0, delta_xc] {
            for da in [-delta_a, 0.0, delta_a] {
                let new_xc = x_c + dx;
                let new_a = (a + da).max(0.001);

                let loss: f64 = x_scaled
                    .iter()
                    .zip(accuracies.iter())
                    .map(|(&x, &y)| {
                        let pred = sigmoid(x, new_xc, new_a, y_min, y_max);
                        (pred - y).powi(2)
                    })
                    .sum();

                if loss < best_loss {
                    best_loss = loss;
                    x_c = new_xc;
                    a = new_a;
                }
            }
        }
    }

    Some((x_c, a))
}

// ============================================================================
// EXPERIMENT RUNNER
// ============================================================================

fn run_experiment_config(
    n: usize,
    l: usize,
    k: usize,
    rho: f64,
    m_values: &[usize],
    n_trials: usize,
    k_clusters: usize,
) -> (Option<f64>, Option<f64>, Vec<f64>, f64) {
    let mut lab = REWALab::new(n, 300000, l, 42);
    lab.generate_clustered_data(k_clusters, rho);
    let delta = lab.measure_gap(500);

    let accuracies: Vec<f64> = m_values
        .iter()
        .map(|&m| {
            let trial_accs: Vec<f64> = (0..n_trials)
                .map(|t| {
                    let enc = lab.encode(m, k, 42 + t as u64);
                    lab.evaluate_retrieval(&enc)
                })
                .collect();
            trial_accs.iter().sum::<f64>() / trial_accs.len() as f64
        })
        .collect();

    let fit_result = fit_critical_point(m_values, &accuracies, delta);
    let (chi_c, width) = match fit_result {
        Some((c, w)) => (Some(c), Some(w)),
        None => (None, None),
    };

    (chi_c, width, accuracies, delta)
}

// ============================================================================
// PLOTTING UTILITIES
// ============================================================================

fn plot_k_dependence(
    k_values: &[usize],
    chi_c_values: &[f64],
    all_data: &[(usize, Vec<f64>, Vec<f64>)], // (K, x_scaled, accuracies)
) -> Result<(), Box<dyn std::error::Error>> {
    // Plot 1: Transition curves for different K
    {
        let root = BitMapBackend::new("investigation_exp1_K_dependence.png", (800, 600))
            .into_drawing_area();
        root.fill(&WHITE)?;

        let x_max = all_data
            .iter()
            .flat_map(|(_, xs, _)| xs.iter())
            .cloned()
            .fold(0.0_f64, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Effect of Hash Functions K on Critical Point", ("sans-serif", 20))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d((1.0_f64..x_max).log_scale(), 0.0..1.0)?;

        chart
            .configure_mesh()
            .x_desc("χ = m·Δ²")
            .y_desc("Accuracy")
            .draw()?;

        let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];
        for (idx, (k, x_scaled, accuracies)) in all_data.iter().enumerate() {
            let color = colors[idx % colors.len()];
            let chi_c = chi_c_values.get(idx).unwrap_or(&0.0);

            let points: Vec<(f64, f64)> = x_scaled
                .iter()
                .zip(accuracies.iter())
                .map(|(&x, &y)| (x.max(1.0), y))
                .collect();

            chart
                .draw_series(LineSeries::new(points.clone(), color))?
                .label(format!("K={} (χ_c={:.1})", k, chi_c))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

            chart.draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, color.filled())),
            )?;
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::LowerRight)
            .draw()?;

        root.present()?;
    }

    // Plot 2: chi_c vs K
    {
        let root =
            BitMapBackend::new("investigation_exp1_chi_vs_K.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        let k_max = *k_values.iter().max().unwrap_or(&10) as f64;
        let chi_max = chi_c_values.iter().cloned().fold(0.0_f64, f64::max) * 1.2;

        let mut chart = ChartBuilder::on(&root)
            .caption("Does C_M increase with K?", ("sans-serif", 20))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..(k_max + 1.0), 0.0..chi_max)?;

        chart
            .configure_mesh()
            .x_desc("Number of Hash Functions K")
            .y_desc("Critical Point χ_c")
            .draw()?;

        let points: Vec<(f64, f64)> = k_values
            .iter()
            .zip(chi_c_values.iter())
            .map(|(&k, &c)| (k as f64, c))
            .collect();

        chart.draw_series(LineSeries::new(points.clone(), &MAGENTA))?;
        chart.draw_series(
            points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 5, MAGENTA.filled())),
        )?;

        root.present()?;
    }

    Ok(())
}

fn plot_n_scaling(
    n_values: &[usize],
    chi_c_values: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("investigation_exp2_N_scaling.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let log_n: Vec<f64> = n_values.iter().map(|&n| (n as f64).ln()).collect();
    let log_n_min = log_n.iter().cloned().fold(f64::INFINITY, f64::min);
    let log_n_max = log_n.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let chi_max = chi_c_values.iter().cloned().fold(0.0_f64, f64::max) * 1.2;

    // Linear fit: chi_c = slope * log(N) + intercept
    let n_pts = log_n.len() as f64;
    let sum_x: f64 = log_n.iter().sum();
    let sum_y: f64 = chi_c_values.iter().sum();
    let sum_xy: f64 = log_n
        .iter()
        .zip(chi_c_values.iter())
        .map(|(&x, &y)| x * y)
        .sum();
    let sum_x2: f64 = log_n.iter().map(|&x| x * x).sum();

    let slope = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n_pts;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("N-Scaling Validation: χ_c ∝ log(N)\nEmpirical C_M = {:.2}", slope),
            ("sans-serif", 18),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(log_n_min * 0.95..log_n_max * 1.05, 0.0..chi_max)?;

    chart
        .configure_mesh()
        .x_desc("log(N)")
        .y_desc("Critical Point χ_c")
        .draw()?;

    // Plot data points
    let points: Vec<(f64, f64)> = log_n
        .iter()
        .zip(chi_c_values.iter())
        .map(|(&x, &y)| (x, y))
        .collect();

    chart.draw_series(
        points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled())),
    )?;

    // Plot fit line
    let fit_points: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let x = log_n_min + (log_n_max - log_n_min) * i as f64 / 99.0;
            (x, slope * x + intercept)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(fit_points, &RED.mix(0.8)))?
        .label(format!("Fit: slope (C_M) = {:.2}", slope))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn plot_rho_dependence(
    rho_values: &[f64],
    chi_c_values: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("investigation_exp3_rho_dependence.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let rho_min = rho_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let rho_max = rho_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let chi_max = chi_c_values.iter().cloned().fold(0.0_f64, f64::max) * 1.2;

    let mut chart = ChartBuilder::on(&root)
        .caption("Universality Check: Is χ_c independent of ρ?", ("sans-serif", 18))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(rho_min * 0.9..rho_max * 1.1, 0.0..chi_max.max(1.0))?;

    chart
        .configure_mesh()
        .x_desc("Signal Strength ρ")
        .y_desc("Critical Point χ_c")
        .draw()?;

    let points: Vec<(f64, f64)> = rho_values
        .iter()
        .zip(chi_c_values.iter())
        .map(|(&r, &c)| (r, c))
        .collect();

    chart.draw_series(LineSeries::new(points.clone(), &GREEN))?;
    chart.draw_series(
        points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, GREEN.filled())),
    )?;

    root.present()?;
    Ok(())
}

fn plot_l_dependence(
    l_values: &[usize],
    chi_c_values: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("investigation_exp4_L_dependence.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let l_min = *l_values.iter().min().unwrap_or(&25) as f64;
    let l_max = *l_values.iter().max().unwrap_or(&200) as f64;
    let chi_max = chi_c_values.iter().cloned().fold(0.0_f64, f64::max) * 1.2;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Does Witness Count L affect Critical Point?",
            ("sans-serif", 18),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(l_min * 0.9..l_max * 1.1, 0.0..chi_max.max(1.0))?;

    chart
        .configure_mesh()
        .x_desc("Witness Count L")
        .y_desc("Critical Point χ_c")
        .draw()?;

    let points: Vec<(f64, f64)> = l_values
        .iter()
        .zip(chi_c_values.iter())
        .map(|(&l, &c)| (l as f64, c))
        .collect();

    chart.draw_series(LineSeries::new(points.clone(), &RGBColor(255, 165, 0)))?; // Orange
    chart.draw_series(
        points
            .iter()
            .map(|&(x, y)| Circle::new((x, y), 5, RGBColor(255, 165, 0).filled())),
    )?;

    root.present()?;
    Ok(())
}

fn plot_precision(
    x_scaled: &[f64],
    accuracies: &[f64],
    std_devs: &[f64],
    chi_c: f64,
    chi_c_err: f64,
    fit_params: (f64, f64, f64, f64),
) -> Result<(), Box<dyn std::error::Error>> {
    let root =
        BitMapBackend::new("investigation_exp5_precision.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = x_scaled.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("High-Precision Transition Analysis", ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min * 0.9..x_max * 1.1, 0.0..1.1)?;

    chart
        .configure_mesh()
        .x_desc("χ = m·Δ²")
        .y_desc("Accuracy")
        .draw()?;

    // Plot error bars and data points
    for (i, (&x, (&y, &err))) in x_scaled
        .iter()
        .zip(accuracies.iter().zip(std_devs.iter()))
        .enumerate()
    {
        let _ = i;
        // Error bar
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, (y - err).max(0.0)), (x, (y + err).min(1.0))],
            &BLUE,
        )))?;
        // Data point
        chart.draw_series(std::iter::once(Circle::new((x, y), 4, BLUE.filled())))?;
    }

    // Plot fit curve
    let (x_c, a, y_min, y_max) = fit_params;
    let fit_points: Vec<(f64, f64)> = (0..200)
        .map(|i| {
            let x = x_min + (x_max - x_min) * i as f64 / 199.0;
            (x, sigmoid(x, x_c, a, y_min, y_max))
        })
        .collect();

    chart
        .draw_series(LineSeries::new(fit_points, &RED))?
        .label(format!("Fit: χ_c = {:.2} ± {:.2}", chi_c, chi_c_err))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

// ============================================================================
// EXPERIMENTS
// ============================================================================

fn experiment_1_k_dependence() -> ExperimentResult {
    println!("\n{}", "=".repeat(60));
    println!(" EXPERIMENT 1: K-Dependence (The C_M Mystery)");
    println!("{}", "=".repeat(60));

    let k_values = vec![2, 3, 4, 6, 8];
    let m_values: Vec<usize> = geomspace(16.0, 4096.0, 15);

    let mut results_k = Vec::new();
    let mut results_chi_c = Vec::new();
    let mut all_data: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();

    for &k in &k_values {
        println!("Testing K={}...", k);
        let (chi_c, _, accs, delta) =
            run_experiment_config(2000, 75, k, 0.4, &m_values, 3, 20);

        if let Some(c) = chi_c {
            results_k.push(k);
            results_chi_c.push(c);
            let x_scaled: Vec<f64> = m_values.iter().map(|&m| m as f64 * delta * delta).collect();
            all_data.push((k, x_scaled, accs));
            println!("  -> χ_c = {:.2}", c);
        }
    }

    // Generate plots
    if let Err(e) = plot_k_dependence(&results_k, &results_chi_c, &all_data) {
        eprintln!("Plot error: {}", e);
    }

    ExperimentResult {
        k_values: Some(results_k),
        n_values: None,
        rho_values: None,
        l_values: None,
        chi_c_values: Some(results_chi_c),
        chi_c: None,
        chi_c_err: None,
    }
}

fn experiment_2_n_scaling() -> ExperimentResult {
    println!("\n{}", "=".repeat(60));
    println!(" EXPERIMENT 2: N-Scaling (The Core Theory Test)");
    println!("{}", "=".repeat(60));

    let n_values = vec![500, 1000, 2000, 4000, 8000];
    let m_values: Vec<usize> = geomspace(16.0, 8192.0, 15);

    let mut results_n = Vec::new();
    let mut results_chi_c = Vec::new();

    for &n in &n_values {
        println!("Testing N={}...", n);
        // Adjust clusters to keep items/cluster constant (~100)
        let k_clusters = (n / 100).max(5);

        let mut lab = REWALab::new(n, 300000, 75, 42);
        lab.generate_clustered_data(k_clusters, 0.4);
        let delta = lab.measure_gap(500);

        let accuracies: Vec<f64> = m_values
            .iter()
            .map(|&m| {
                let trial_accs: Vec<f64> = (0..3)
                    .map(|t| {
                        let enc = lab.encode(m, 4, 42 + t);
                        lab.evaluate_retrieval(&enc)
                    })
                    .collect();
                trial_accs.iter().sum::<f64>() / trial_accs.len() as f64
            })
            .collect();

        if let Some((chi_c, _)) = fit_critical_point(&m_values, &accuracies, delta) {
            results_n.push(n);
            results_chi_c.push(chi_c);
            println!("  -> χ_c = {:.2}", chi_c);
        }
    }

    // Generate plot
    if let Err(e) = plot_n_scaling(&results_n, &results_chi_c) {
        eprintln!("Plot error: {}", e);
    }

    ExperimentResult {
        k_values: None,
        n_values: Some(results_n),
        rho_values: None,
        l_values: None,
        chi_c_values: Some(results_chi_c),
        chi_c: None,
        chi_c_err: None,
    }
}

fn experiment_3_rho_fine() -> ExperimentResult {
    println!("\n{}", "=".repeat(60));
    println!(" EXPERIMENT 3: Rho-Dependence (Universality Check)");
    println!("{}", "=".repeat(60));

    let rho_values = vec![0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6];
    let m_values: Vec<usize> = geomspace(16.0, 8192.0, 15);

    let mut results_rho = Vec::new();
    let mut results_chi_c = Vec::new();

    for &rho in &rho_values {
        println!("Testing rho={}...", rho);
        let (chi_c, _, _, _) = run_experiment_config(2000, 75, 4, rho, &m_values, 3, 20);

        if let Some(c) = chi_c {
            results_rho.push(rho);
            results_chi_c.push(c);
            println!("  -> χ_c = {:.2}", c);
        }
    }

    // Generate plot
    if let Err(e) = plot_rho_dependence(&results_rho, &results_chi_c) {
        eprintln!("Plot error: {}", e);
    }

    ExperimentResult {
        k_values: None,
        n_values: None,
        rho_values: Some(results_rho),
        l_values: None,
        chi_c_values: Some(results_chi_c),
        chi_c: None,
        chi_c_err: None,
    }
}

fn experiment_4_l_dependence() -> ExperimentResult {
    println!("\n{}", "=".repeat(60));
    println!(" EXPERIMENT 4: L-Dependence (Witness Count)");
    println!("{}", "=".repeat(60));

    let l_values = vec![25, 50, 75, 100, 150, 200];
    let m_values: Vec<usize> = geomspace(16.0, 8192.0, 15);

    let mut results_l = Vec::new();
    let mut results_chi_c = Vec::new();

    for &l in &l_values {
        println!("Testing L={}...", l);
        let (chi_c, _, _, _) = run_experiment_config(2000, l, 4, 0.4, &m_values, 3, 20);

        if let Some(c) = chi_c {
            results_l.push(l);
            results_chi_c.push(c);
            println!("  -> χ_c = {:.2}", c);
        }
    }

    // Generate plot
    if let Err(e) = plot_l_dependence(&results_l, &results_chi_c) {
        eprintln!("Plot error: {}", e);
    }

    ExperimentResult {
        k_values: None,
        n_values: None,
        rho_values: None,
        l_values: Some(results_l),
        chi_c_values: Some(results_chi_c),
        chi_c: None,
        chi_c_err: None,
    }
}

fn experiment_5_precision() -> ExperimentResult {
    println!("\n{}", "=".repeat(60));
    println!(" EXPERIMENT 5: Precision (Transition Sharpness)");
    println!("{}", "=".repeat(60));

    // Dense sampling around expected critical point
    let m_dense: Vec<usize> = linspace(50.0, 400.0, 30);
    let n_trials_high = 20;

    println!("Running dense sweep with {} trials...", n_trials_high);
    let mut lab = REWALab::new(2000, 300000, 75, 42);
    lab.generate_clustered_data(20, 0.4);
    let delta = lab.measure_gap(1000);

    let mut accuracies = Vec::new();
    let mut std_devs = Vec::new();

    for &m in &m_dense {
        let trial_accs: Vec<f64> = (0..n_trials_high)
            .map(|t| {
                let enc = lab.encode(m, 4, 100 + t as u64);
                lab.evaluate_retrieval(&enc)
            })
            .collect();

        let mean = trial_accs.iter().sum::<f64>() / trial_accs.len() as f64;
        let variance = trial_accs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / trial_accs.len() as f64;
        let std_dev = variance.sqrt();

        accuracies.push(mean);
        std_devs.push(std_dev);
    }

    let x_scaled: Vec<f64> = m_dense.iter().map(|&m| m as f64 * delta * delta).collect();

    // Fit with weights based on std_devs
    let fit_result = fit_critical_point(&m_dense, &accuracies, delta);

    let (chi_c, chi_c_err, fit_params) = match fit_result {
        Some((c, a)) => {
            // Estimate error from fit quality
            let residuals: f64 = x_scaled
                .iter()
                .zip(accuracies.iter())
                .map(|(&x, &y)| {
                    let pred = sigmoid(x, c, a, 0.05, 0.95);
                    (pred - y).powi(2)
                })
                .sum();
            let err = (residuals / x_scaled.len() as f64).sqrt() * c * 0.1;
            (c, err, (c, a, 0.05, 0.95))
        }
        None => (0.0, 0.0, (0.0, 0.05, 0.0, 1.0)),
    };

    println!("  -> χ_c = {:.4} ± {:.4}", chi_c, chi_c_err);

    // Generate plot
    if let Err(e) = plot_precision(&x_scaled, &accuracies, &std_devs, chi_c, chi_c_err, fit_params)
    {
        eprintln!("Plot error: {}", e);
    }

    ExperimentResult {
        k_values: None,
        n_values: None,
        rho_values: None,
        l_values: None,
        chi_c_values: None,
        chi_c: Some(chi_c),
        chi_c_err: Some(chi_c_err),
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn geomspace(start: f64, end: f64, n: usize) -> Vec<usize> {
    let log_start = start.ln();
    let log_end = end.ln();
    (0..n)
        .map(|i| {
            let log_val = log_start + (log_end - log_start) * i as f64 / (n - 1) as f64;
            log_val.exp().round() as usize
        })
        .collect()
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| (start + (end - start) * i as f64 / (n - 1) as f64).round() as usize)
        .collect()
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("UNIFIED INVESTIGATION SUITE: Probing the Rough Edges of Semantic Thermodynamics");
    println!("================================================================================\n");

    let exp1 = experiment_1_k_dependence();
    let exp2 = experiment_2_n_scaling();
    let exp3 = experiment_3_rho_fine();
    let exp4 = experiment_4_l_dependence();
    let exp5 = experiment_5_precision();

    let all_results = AllResults {
        exp1,
        exp2,
        exp3,
        exp4,
        exp5,
    };

    // Save results to JSON
    let json = serde_json::to_string_pretty(&all_results).unwrap();
    let mut file = File::create("investigation_results.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();

    println!("\n{}", "=".repeat(80));
    println!(" INVESTIGATION COMPLETE");
    println!("{}", "=".repeat(80));
    println!("Generated 5 figures and investigation_results.json");
}
