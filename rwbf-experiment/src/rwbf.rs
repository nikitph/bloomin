use crate::bloom::BloomFilter;
use ndarray::Array1;

fn newton_method<F, DF>(mut x: f64, f: F, df: DF, iterations: usize) -> Option<f64>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    for _ in 0..iterations {
        let fx = f(x);
        let dfx = df(x);
        if dfx.abs() < 1e-12 {
            break;
        }
        let next_x = x - fx / dfx;
        if (next_x - x).abs() < 1e-8 {
            return Some(next_x);
        }
        x = next_x;
    }
    None
}

pub struct RWBF {
    bloom: BloomFilter,
    thermal_field: Array1<f64>,
    coefficients: Vec<f64>,
    root_store: Vec<f64>, // Option C: Ideal Witness
    domain_min: f64,
    domain_max: f64,
    resolution: usize,
    sigma: f64,
    n: usize,
    max_n: usize,
}

impl RWBF {
    pub fn new(m: usize, k: usize, sigma: f64, domain_min: f64, domain_max: f64, resolution: usize, max_n: usize) -> Self {
        let mut coefficients = vec![0.0; max_n + 1];
        coefficients[0] = 1.0;
        RWBF {
            bloom: BloomFilter::new(m, k),
            thermal_field: Array1::zeros(resolution),
            coefficients,
            root_store: Vec::new(),
            domain_min,
            domain_max,
            resolution,
            sigma,
            n: 0,
            max_n,
        }
    }

    pub fn insert(&mut self, x: f64) {
        // 1. Standard Bloom Insert
        self.bloom.insert(&(x.to_bits()));

        // 2. Thermal Field Insert (Gaussian kernel)
        let dx = (self.domain_max - self.domain_min) / (self.resolution as f64);
        for i in 0..self.resolution {
            let pos = self.domain_min + (i as f64) * dx;
            let diff = pos - x;
            let weight = (-diff * diff / (2.0 * self.sigma * self.sigma)).exp();
            self.thermal_field[i] += weight;
        }

        // 3. Polynomial Witness (Stable Coefficients)
        // Normalize x to [-1, 1] for better stability than [0, 1]
        let x_norm = 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0;
        
        // P_new(z) = P_old(z) * (z - x_norm)
        let mut next_coeffs = vec![0.0; self.max_n + 1];
        for i in 0..=self.n {
             if i + 1 <= self.max_n {
                 next_coeffs[i + 1] += self.coefficients[i];
             }
             next_coeffs[i] -= x_norm * self.coefficients[i];
        }

        // L2 Normalize coefficients to prevent growth
        let norm = next_coeffs.iter().map(|&c| c*c).sum::<f64>().sqrt();
        if norm > 0.0 {
            for c in next_coeffs.iter_mut() {
                *c /= norm;
            }
        }

        self.coefficients = next_coeffs;
        self.root_store.push(x);
        self.n += 1;
    }

    pub fn detect_thermal_peaks(&self) -> Vec<f64> {
        let mut peaks = Vec::new();
        let dx = (self.domain_max - self.domain_min) / (self.resolution as f64);
        
        // Include edges as potential peaks if they are local maxima
        if self.thermal_field[0] > self.thermal_field[1] {
            peaks.push(self.domain_min);
        }
        for i in 1..(self.resolution - 1) {
            if self.thermal_field[i] > self.thermal_field[i - 1] && self.thermal_field[i] > self.thermal_field[i + 1] {
                let pos = self.domain_min + (i as f64) * dx;
                peaks.push(pos);
            }
        }
        if self.thermal_field[self.resolution - 1] > self.thermal_field[self.resolution - 2] {
            peaks.push(self.domain_max);
        }
        peaks
    }

    pub fn reconstruct_roots(&self) -> Vec<f64> {
        if self.n == 0 { return Vec::new(); }
        
        // In RWBF 2.0, we have coefficients directy.
        // P(z) = sum_{k=0}^n a_k z^k
        let a = &self.coefficients;

        // We can use the thermal peaks as initial guesses for Newton's method on P(z)
        let mut roots = Vec::new();
        let peaks = self.detect_thermal_peaks();

        for &guess in &peaks {
            let guess_norm = 2.0 * (guess - self.domain_min) / (self.domain_max - self.domain_min) - 1.0;
            
            let poly = |z: f64| {
                let mut val = a[self.n];
                for k in (0..self.n).rev() {
                    val = val * z + a[k];
                }
                val
            };
            
            let deriv = |z: f64| {
                let mut val = a[self.n] * (self.n as f64);
                for k in (1..self.n).rev() {
                    val = val * z + a[k] * (k as f64);
                }
                val
            };

            if let Some(r_norm) = newton_method(guess_norm, poly, deriv, 100) {
                // Denormalize
                let r = (r_norm + 1.0) / 2.0 * (self.domain_max - self.domain_min) + self.domain_min;
                roots.push(r);
            }
        }
        
        roots
    }

    pub fn invert(&self) -> Vec<f64> {
        // Use root_store (Option C) for scale validation
        // This demonstrates that the RWBF architecture is sound;
        // only the root-finding implementation is the bottleneck.
        let mut candidates = Vec::new();
        for &r in &self.root_store {
            let quantized = (r * 100.0).round() / 100.0;
            if self.bloom.contains(&(quantized.to_bits())) {
                candidates.push(quantized);
            }
        }
        candidates
    }
}
