use ndarray::{Array2, Axis};
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};
use std::time::Instant;

fn fwht(a: &mut [f32]) {
    let n = a.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

fn main() {
    let n_vectors = 10000;
    let d = 1024;
    let m = 128;
    
    println!("--- Rust Benchmark ---");
    println!("Encoding {} vectors of dimension {} -> {}", n_vectors, d, m);
    
    // Data Generation
    let mut rng = thread_rng();
    let x_data: Vec<f32> = (0..n_vectors * d).map(|_| rng.sample(StandardNormal)).collect();
    let x = Array2::from_shape_vec((n_vectors, d), x_data).unwrap();
    
    let g_data: Vec<f32> = (0..d * m).map(|_| rng.sample(StandardNormal)).collect();
    let g = Array2::from_shape_vec((d, m), g_data).unwrap();
    
    let d_vec: Vec<f32> = (0..d).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect();
    
    // Warmup
    let _ = x.dot(&g);
    
    // Benchmark Random Projection
    let start = Instant::now();
    let _res_rp = x.dot(&g);
    let duration_rp = start.elapsed();
    println!("Random Projection: {:.2?}", duration_rp);
    
    // Benchmark Witness Polar
    // We do this row by row for the FWHT as it's typically applied per vector
    // Or we can implement a batched version. For fair comparison with Python loop, let's do row-wise but optimized.
    // Actually, the Python version did `fwht(X * D)` which implies batched.
    // Our recursive FWHT is single vector. Let's wrap it.
    
    let start = Instant::now();
    let mut x_polar = x.clone();
    
    // 1. Diagonal Flip
    for mut row in x_polar.axis_iter_mut(Axis(0)) {
        for i in 0..d {
            row[i] *= d_vec[i];
        }
    }
    
    // 2. FWHT
    // We can parallelize this with rayon later if needed, but let's stick to single thread for fair comparison with single-threaded BLAS (usually).
    // Actually BLAS is multi-threaded. We should probably allow Rayon for FWHT to be fair?
    // For now, sequential.
    for mut row in x_polar.axis_iter_mut(Axis(0)) {
        let slice = row.as_slice_mut().unwrap();
        fwht(slice);
    }
    
    // 3. Select m columns
    // In Rust ndarray, slicing is a view, but for "result" we might want to copy.
    // Let's just take the view to be charitable, or copy to be realistic.
    let _res_wp = x_polar.slice(ndarray::s![.., ..m]).to_owned();
    
    let duration_wp = start.elapsed();
    println!("Witness-Polar:     {:.2?}", duration_wp);
    
    let speedup = duration_rp.as_secs_f64() / duration_wp.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);
}
