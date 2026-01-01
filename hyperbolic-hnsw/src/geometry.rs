use ndarray::{Array1, ArrayView1};
use std::f64;

pub struct PoincareBall {
    pub dim: usize,
    pub c: f64,
    pub eps: f64,
}

impl PoincareBall {
    pub fn new(dim: usize, c: f64) -> Self {
        Self {
            dim,
            c,
            eps: 1e-6,
        }
    }

    /// Möbius addition: x ⊕ y
    pub fn mobius_add(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let x_dot_x = x.dot(x);
        let y_dot_y = y.dot(y);
        let x_dot_y = x.dot(y);
        
        let c = self.c;
        let numerator = (1.0 + 2.0 * c * x_dot_y + c * y_dot_y) * x 
                      + (1.0 - c * x_dot_x) * y;
        
        let denominator = 1.0 + 2.0 * c * x_dot_y + c.powi(2) * x_dot_x * y_dot_y;
        
        numerator / (denominator + self.eps)
    }

    /// Möbius subtraction: x ⊖ y = x ⊕ (-y)
    pub fn mobius_sub(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let neg_y = -y;
        self.mobius_add(x, &neg_y.view())
    }

    /// Hyperbolic distance
    pub fn distance(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        let diff = self.mobius_sub(x, y);
        let diff_norm = diff.dot(&diff).sqrt();
        
        // Clip to valid range
        let max_val = 1.0 - self.eps;
        let clipped_norm = if diff_norm > max_val { max_val } else { diff_norm };
        
        let sqrt_c = self.c.sqrt();
        (2.0 / sqrt_c) * (sqrt_c * clipped_norm).atanh()
    }

    /// Project point to interior of Poincaré ball
    pub fn project_to_ball(&self, x: Array1<f64>) -> Array1<f64> {
        let norm = x.dot(&x).sqrt();
        let max_norm = (1.0 / self.c.sqrt()) - self.eps;
        
        if norm >= max_norm {
            (max_norm / norm) * x
        } else {
            x
        }
    }

    /// Conformal factor λ_x = 2 / (1 - c||x||²)
    pub fn conformal_factor(&self, x: &ArrayView1<f64>) -> f64 {
        let x_norm_sq = x.dot(x);
        2.0 / (1.0 - self.c * x_norm_sq + self.eps)
    }

    /// Norm of tangent vector v at x: ||v||_x = ||v|| / λ_x
    pub fn norm_tangent(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> f64 {
        let lambda_x = self.conformal_factor(x);
        let v_norm = v.dot(v).sqrt();
        v_norm / lambda_x
    }

    /// Exponential map: TₓB^d → B^d
    pub fn exp_map(&self, x: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        let v_norm_x = self.norm_tangent(x, v);
        
        if v_norm_x < self.eps {
            return x.to_owned();
        }
        
        let sqrt_c = self.c.sqrt();
        let scale = (sqrt_c * v_norm_x / 2.0).tanh() / (sqrt_c * v_norm_x);
        
        let scaled_v = scale * v;
        let result = self.mobius_add(x, &scaled_v.view());
        
        self.project_to_ball(result)
    }

    /// Logarithmic map: B^d → TₓB^d
    pub fn log_map(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let neg_x = -x;
        let diff = self.mobius_add(&neg_x.view(), y);
        let diff_norm = diff.dot(&diff).sqrt();
        
        if diff_norm < self.eps {
            return Array1::zeros(self.dim);
        }
        
        let sqrt_c = self.c.sqrt();
        let scale = (2.0 / sqrt_c) * (sqrt_c * diff_norm).atanh() / diff_norm;
        
        scale * diff
    }

    /// Parallel transport vector v from TₓM to TᵧM
    pub fn parallel_transport(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>, v: &ArrayView1<f64>) -> Array1<f64> {
        // If x is close to y, no transport needed
        // (Using simplified allclose check)
        if (&(x - y)).mapv(|a| a.abs()).sum() < self.eps {
             return v.to_owned();
        }

        let log_xy = self.log_map(x, y);
        let log_yx = self.log_map(y, x);
        
        let lambda_x = self.conformal_factor(x);
        let lambda_y = self.conformal_factor(y);
        
        let log_xy_norm_sq = log_xy.dot(&log_xy);
        
        if log_xy_norm_sq < self.eps {
            return v * (lambda_x / lambda_y);
        }
        
        let projection = log_xy.dot(v) / log_xy_norm_sq;
        let transported = v - projection * (&log_xy + &log_yx);
        
        (lambda_x / lambda_y) * transported
    }
}
