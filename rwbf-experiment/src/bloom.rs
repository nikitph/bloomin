use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct BloomFilter {
    bits: Vec<bool>,
    m: usize,
    k: usize,
}

impl BloomFilter {
    pub fn new(m: usize, k: usize) -> Self {
        BloomFilter {
            bits: vec![false; m],
            m,
            k,
        }
    }

    pub fn insert<T: Hash>(&mut self, item: &T) {
        for i in 0..self.k {
            let hash = self.hash(item, i);
            self.bits[hash % self.m] = true;
        }
    }

    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        for i in 0..self.k {
            let hash = self.hash(item, i);
            if !self.bits[hash % self.m] {
                return false;
            }
        }
        true
    }

    fn hash<T: Hash>(&self, item: &T, i: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        i.hash(&mut hasher);
        hasher.finish() as usize
    }

    pub fn attempt_invert(&self, domain: &[f64]) -> Vec<f64> {
        let mut candidates = Vec::new();
        for &x in domain {
            // Mapping f64 to a hashable representation for standard Bloom
            let bits = x.to_bits();
            if self.contains(&bits) {
                candidates.push(x);
            }
        }
        candidates
    }
}
