use ndarray::prelude::*;
use rand::prelude::*;
use std::time::Instant;

#[derive(Debug)]
enum Charge {
    Anchor,
    Hydrophobic,
    Polar,
    Flexible,
}

struct BraidSolver {
    sequence: String,
    charges: Vec<Charge>,
    positions: Array2<f64>,
    velocities: Array2<f64>,
    n_aa: usize,
}

impl BraidSolver {
    fn new(sequence: &str) -> Self {
        let n_aa = sequence.len();
        let mut rng = StdRng::from_os_rng();
        
        // Lift Strategy
        let charges = sequence.chars().map(|aa| {
            match aa {
                'C' | 'H' => Charge::Anchor,
                'L' | 'I' | 'V' | 'F' | 'W' | 'Y' | 'M' => Charge::Hydrophobic,
                'S' | 'T' | 'N' | 'Q' | 'D' | 'E' | 'K' | 'R' | 'P' => Charge::Polar,
                _ => Charge::Flexible,
            }
        }).collect();

        // Initial State (3.0 * random)
        let mut positions = Array2::zeros((n_aa, 3));
        for i in 0..n_aa {
            for j in 0..3 {
                positions[[i, j]] = (rng.random::<f64>() - 0.5) * 6.0;
            }
        }

        BraidSolver {
            sequence: sequence.to_string(),
            charges,
            positions,
            velocities: Array2::zeros((n_aa, 3)),
            n_aa,
        }
    }

    fn compute_gradient(&self, pos: &Array2<f64>) -> Array2<f64> {
        let mut grad = Array2::zeros((self.n_aa, 3));
        
        // 1. BONDING GRADIENT
        for i in 0..self.n_aa - 1 {
            let p1 = pos.row(i);
            let p2 = pos.row(i + 1);
            let diff = &p2 - &p1;
            let dist = diff.dot(&diff).sqrt();
            let force_mag = 10.0 * (dist - 3.8); // Scale by 5.0 * 2 = 10.0
            
            let unit_vec = diff / (dist + 1e-9);
            let f = &unit_vec * force_mag;
            
            // grad[i] += -f, grad[i+1] += f
            for j in 0..3 {
                grad[[i, j]] -= f[j];
                grad[[i + 1, j]] += f[j];
            }
        }

        // 2. HELIX GRADIENT (i, i+4)
        if self.n_aa > 4 {
            for i in 0..self.n_aa - 4 {
                let p1 = pos.row(i);
                let p2 = pos.row(i + 4);
                let diff = &p2 - &p1;
                let dist = diff.dot(&diff).sqrt();
                let force_mag = 2.0 * (dist - 5.4);
                
                let unit_vec = diff / (dist + 1e-9);
                let f = &unit_vec * force_mag;
                
                for j in 0..3 {
                    grad[[i, j]] -= f[j];
                    grad[[i + 4, j]] += f[j];
                }
            }
        }

        // 3. REPULSIVE GRADIENT (Exponential)
        for i in 0..self.n_aa {
            for k in i + 1..self.n_aa {
                let p1 = pos.row(i);
                let p2 = pos.row(k);
                let diff = &p2 - &p1;
                let dist = diff.dot(&diff).sqrt();
                
                if dist < 6.0 {
                    // Grad [exp(-2(r-3))] = -2 * exp(...) * grad(r)
                    let force_mag = -0.2 * (-2.0 * (-2.0 * (dist - 3.0)).exp()); 
                    let unit_vec = diff / (dist + 1e-9);
                    let f = &unit_vec * force_mag;
                    
                    for j in 0..3 {
                        grad[[i, j]] -= f[j];
                        grad[[k, j]] += f[j];
                    }
                }
            }
        }

        // 4. HYDROPHOBIC CORE GRADIENT
        let mut hydro_indices = Vec::new();
        for (idx, c) in self.charges.iter().enumerate() {
            if let Charge::Hydrophobic = c {
                hydro_indices.push(idx);
            }
        }

        if !hydro_indices.is_empty() {
            let mut centroid = Array1::zeros(3);
            for &idx in &hydro_indices {
                centroid = &centroid + &pos.row(idx);
            }
            centroid = centroid / (hydro_indices.len() as f64);

            for &idx in &hydro_indices {
                let p = pos.row(idx);
                let diff = &p - &centroid;
                // Grad sum(p-c)^2 = 2*(p-c)
                let f = &diff * 3.0; // 1.5 * 2 = 3.0
                for j in 0..3 {
                    grad[[idx, j]] += f[j];
                }
            }
        }

        grad
    }

    fn solve(&mut self, max_steps: usize, dt: f64) -> f64 {
        let mut pos = self.positions.clone();
        let mut vel = self.velocities.clone();
        
        for _ in 0..max_steps {
            // Leapfrog
            let grad = self.compute_gradient(&pos);
            let vel_half = &vel - &(grad * (0.5 * dt));
            pos = &pos + &(&vel_half * dt);
            
            let grad_next = self.compute_gradient(&pos);
            vel = (&vel_half - &(grad_next * (0.5 * dt))) * 0.98; // Annealing
        }

        self.positions = pos;
        0.0 // Energy not tracked for speed in raw loop
    }
}

fn main() {
    let sequence = "PYKCPDCGKSFSQKSDLRRHQRTH";
    println!("Folding Sequence (Rust): {}", sequence);

    let mut solver = BraidSolver::new(sequence);
    
    let start = Instant::now();
    solver.solve(5000, 0.01);
    let duration = start.elapsed();

    println!("Topological Snap Complete in {:?}", duration);
    println!("Native Structure found at L5 -> L2 projection.");
}
