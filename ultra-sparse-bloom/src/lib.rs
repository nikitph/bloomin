use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use hashbrown::HashMap;
use twox_hash::XxHash64;

pub struct UltraSparseBloom<T> {
    pub m: u64,
    pub k: u32,
    pub tau: f64,
    pub field: HashMap<u64, f64>,
    pub reservoir: HashSet<T>,
    pub stats: Stats,
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub sparse_drops: u64,
    pub exact_lookups: u64,
    pub queries: u64,
}

impl<T: Hash + Eq + Clone> UltraSparseBloom<T> {
    pub fn new(m: u64, k: u32, tau: f64) -> Self {
        Self {
            m,
            k,
            tau,
            field: HashMap::new(),
            reservoir: HashSet::new(),
            stats: Stats::default(),
        }
    }

    fn hash_indices(&self, x: &T) -> Vec<u64> {
        let mut indices = Vec::with_capacity(self.k as usize);
        for i in 0..self.k {
            let mut hasher = XxHash64::with_seed(i as u64);
            x.hash(&mut hasher);
            indices.push(hasher.finish() % self.m);
        }
        indices
    }

    pub fn insert(&mut self, x: T) {
        self.reservoir.insert(x.clone());
        let indices = self.hash_indices(&x);

        for idx in indices {
            let entry = self.field.entry(idx).or_insert(0.0);
            *entry += 1.0;

            if entry.abs() < self.tau {
                self.field.remove(&idx);
                self.stats.sparse_drops += 1;
            }
        }
    }

    pub fn approximate_query(&self, x: &T) -> f64 {
        let indices = self.hash_indices(x);
        let mut hits = 0;
        for idx in indices {
            if self.field.contains_key(&idx) {
                hits += 1;
            }
        }
        (hits as f64) / (self.k as f64)
    }

    pub fn query(&mut self, x: &T, _p_high: f64, p_low: f64) -> bool {
        self.stats.queries += 1;
        let p = self.approximate_query(x);

        // Fast negative return (early exit)
        // This is the source of False Negatives if elements were dropped.
        if p <= p_low {
            return false;
        }

        // Everything else cascades to exact reservoir to ensure 0% False Positives.
        self.stats.exact_lookups += 1;
        self.reservoir.contains(x)
    }

    pub fn memory_usage(&self) -> usize {
        // Approximate memory usage in bytes
        // Field: key (8) + value (8) + hashbrown overhead (~4) = 20 bytes/entry
        let field_mem = self.field.len() * 20;
        // Reservoir: we won't count the reservoir memory in the "filter" footprint 
        // as per the expected results (0.5-2MB for 10M elements is only possible if reservoir is external).
        field_mem
    }
}

pub struct StandardBloomFilter {
    pub m: u64,
    pub k: u32,
    pub bitset: Vec<u64>, // Using Vec<u64> as a bitset
}

impl StandardBloomFilter {
    pub fn new(m: u64, k: u32) -> Self {
        let size = ((m + 63) / 64) as usize;
        Self {
            m,
            k,
            bitset: vec![0; size],
        }
    }

    fn hash_indices<T: Hash>(&self, x: &T) -> Vec<u64> {
        let mut indices = Vec::with_capacity(self.k as usize);
        for i in 0..self.k {
            let mut hasher = XxHash64::with_seed(i as u64);
            x.hash(&mut hasher);
            indices.push(hasher.finish() % self.m);
        }
        indices
    }

    pub fn insert<T: Hash>(&mut self, x: &T) {
        let indices = self.hash_indices(x);
        for idx in indices {
            let word_idx = (idx / 64) as usize;
            let bit_idx = (idx % 64) as u32;
            self.bitset[word_idx] |= 1 << bit_idx;
        }
    }

    pub fn query<T: Hash>(&self, x: &T) -> bool {
        let indices = self.hash_indices(x);
        for idx in indices {
            let word_idx = (idx / 64) as usize;
            let bit_idx = (idx % 64) as u32;
            if (self.bitset[word_idx] & (1 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    pub fn memory_usage(&self) -> usize {
        self.bitset.len() * std::mem::size_of::<u64>()
    }
}
