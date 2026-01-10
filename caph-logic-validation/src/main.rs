use std::collections::HashMap;
use std::time::Instant;
use rand::prelude::*;
use rand::rngs::StdRng;
use hdrhistogram::Histogram;
use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;

// --- 1. CAPH Implementation (Simulating SIMD Groups) ---
mod caph {
    use super::*;

    const GROUP_WIDTH: usize = 16;
    const PROBE_BUDGET: usize = 2; // Strict budget
    const EMPTY: u8 = 0b1000_0000;
    const OVERFLOW_SENTINEL: u8 = 0b1000_0001; 

    pub struct CaphMap<K, V> {
        ctrl: Vec<u8>,
        slots: Vec<Option<(K, V)>>,
        stash: HashMap<K, V>,
        capacity: usize,
    }

    impl<K: Hash + Eq + Clone, V: Clone> CaphMap<K, V> {
        pub fn new(capacity: usize) -> Self {
            let cap = (capacity + GROUP_WIDTH - 1) & !(GROUP_WIDTH - 1);
            Self {
                ctrl: vec![EMPTY; cap + GROUP_WIDTH],
                slots: vec![None; cap],
                stash: HashMap::new(),
                capacity: cap,
            }
        }

        #[inline(always)]
        fn hash(&self, key: &K) -> (usize, u8) {
            let mut hasher = FxHasher::default();
            key.hash(&mut hasher);
            let h = hasher.finish();
            ((h as usize) % self.capacity, (h & 0x7F) as u8)
        }

        pub fn insert(&mut self, key: K, value: V) {
            let (h1, h2) = self.hash(&key);
            
            for i in 0..PROBE_BUDGET {
                let start = (h1 + i * GROUP_WIDTH) % self.capacity;
                // Since ctrl has GROUP_WIDTH padding, start..start+GROUP_WIDTH is always safe
                let group = &self.ctrl[start..start + GROUP_WIDTH];
                
                for (offset, &c) in group.iter().enumerate() {
                    if c == EMPTY {
                        let idx = (start + offset) % self.capacity;
                        self.ctrl[idx] = h2;
                        self.slots[idx] = Some((key, value));
                        return;
                    }
                    if c == h2 {
                        let idx = (start + offset) % self.capacity;
                        if let Some((k, _)) = &self.slots[idx] {
                            if *k == key {
                                self.slots[idx] = Some((key, value));
                                return;
                            }
                        }
                    }
                }
            }

            self.ctrl[h1 % self.capacity] = OVERFLOW_SENTINEL;
            self.stash.insert(key, value);
        }

        #[inline(always)]
        pub fn get(&self, key: &K) -> Option<&V> {
            let (h1, h2) = self.hash(key);

            for i in 0..PROBE_BUDGET {
                let start = (h1 + i * GROUP_WIDTH) % self.capacity;
                let group = &self.ctrl[start..start + GROUP_WIDTH];
                
                for (offset, &c) in group.iter().enumerate() {
                    if c == h2 {
                        let idx = (start + offset) % self.capacity;
                        if let Some((k, v)) = &self.slots[idx] {
                            if k == key { return Some(v); }
                        }
                    }
                    if c == EMPTY { return None; }
                    if c == OVERFLOW_SENTINEL { break; }
                }
            }
            
            self.stash.get(key)
        }
    }
}

fn main() {
    let num_items = 1_000_000;
    let capacity = (num_items as f64 / 0.95) as usize; 
    
    println!("--- SETUP ---");
    println!("Items: {}", num_items);
    println!("Load Factor: 95%");
    
    let mut rng = StdRng::seed_from_u64(42);
    let keys: Vec<u64> = (0..num_items as u64).map(|_| rng.random()).collect();
    
    println!("\n--- Standard Swiss Table (std::collections::HashMap) ---");
    let mut swiss = HashMap::with_capacity_and_hasher(capacity, rustc_hash::FxBuildHasher::default());
    let mut hist_swiss = Histogram::<u64>::new(3).unwrap();

    for &k in &keys { swiss.insert(k, k); }

    for &k in &keys {
        let start = Instant::now();
        let _ = swiss.get(&k);
        let duration = start.elapsed().as_nanos() as u64;
        hist_swiss.record(duration).unwrap();
    }

    println!("\n--- CAPH (Phase VI) ---");
    let mut caph = caph::CaphMap::new(capacity);
    let mut hist_caph = Histogram::<u64>::new(3).unwrap();

    for &k in &keys { caph.insert(k, k); }

    for &k in &keys {
        let start = Instant::now();
        let _ = caph.get(&k);
        let duration = start.elapsed().as_nanos() as u64;
        hist_caph.record(duration).unwrap();
    }

    println!("\n--- RESULTS (Latency in ns) ---");
    println!("{:<15} {:<10} {:<10} {:<10} {:<10}", "Structure", "Mean", "P99", "P99.9", "Max");
    
    println!("{:<15} {:<10.2} {:<10} {:<10} {:<10}", 
        "Swiss Table", 
        hist_swiss.mean(), 
        hist_swiss.value_at_quantile(0.99),
        hist_swiss.value_at_quantile(0.999),
        hist_swiss.max()
    );

    println!("{:<15} {:<10.2} {:<10} {:<10} {:<10}", 
        "CAPH", 
        hist_caph.mean(), 
        hist_caph.value_at_quantile(0.99),
        hist_caph.value_at_quantile(0.999),
        hist_caph.max()
    );

    println!("\n--- ANALYSIS ---");
    if hist_caph.value_at_quantile(0.999) <= hist_swiss.value_at_quantile(0.999) + 50 {
        println!("✅ VALIDATION SUCCESS: CAPH demonstrates tail latency dominance or parity.");
    } else {
        println!("❌ VALIDATION FAIL: Tail latencies are still significantly worse.");
    }
}
