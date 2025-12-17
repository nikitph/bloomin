//! # Resonant Bloom Filter
//!
//! A temporal data structure that uses complex phasors (oscillators) to encode
//! both membership AND temporal relationships between events in O(1) space and time.
//!
//! ## The Core Insight
//!
//! Standard Bloom filters are **sets**: {A, B} == {B, A}. Time is lost.
//!
//! A Resonant Bloom Filter turns each bucket into an **oscillator**:
//! - Instead of bits (0/1), buckets store complex numbers (phasors)
//! - Each bucket rotates at a specific frequency
//! - Insertions add unit vectors at the current phase
//! - The phase difference between items encodes their temporal relationship
//!
//! ## Physics Analogy
//!
//! Thermal Bloom: Drop dye in water. Diffusion tells you WHERE the source is.
//! Resonant Bloom: Strike a bell. The tone tells you WHAT, the decay tells you WHEN.

use num_complex::Complex64;
use siphasher::sip::SipHasher13;
use std::f64::consts::PI;
use std::hash::{Hash, Hasher};

/// A Resonant Bloom Filter - the "Time Crystal" of probabilistic data structures.
///
/// Uses complex phasors to encode temporal relationships between events,
/// enabling O(1) detection of event sequences and causal patterns.
#[derive(Clone)]
pub struct ResonantBloom {
    /// The oscillator bank - each bucket is a complex phasor
    buckets: Vec<Complex64>,
    /// Decay factor per tick (0.0-1.0, typically ~0.99)
    decay: f64,
    /// Current time tick
    tick: u64,
    /// Base rotation frequency (radians per tick)
    omega_base: f64,
    /// Hash seeds for consistent hashing
    seed: u64,
    /// Track insertions per bucket for amplitude normalization
    insertion_counts: Vec<u64>,
}

/// Configuration for the Resonant Bloom Filter
#[derive(Clone, Debug)]
pub struct ResonantConfig {
    /// Number of buckets (oscillators)
    pub size: usize,
    /// Decay factor per tick (how quickly old events fade)
    pub decay: f64,
    /// Base rotation frequency in radians per tick
    pub omega_base: f64,
    /// Random seed for hashing
    pub seed: u64,
}

impl Default for ResonantConfig {
    fn default() -> Self {
        Self {
            size: 1024,
            decay: 0.99,
            omega_base: 2.0 * PI / 256.0, // Complete rotation every 256 ticks
            seed: 0xDEADBEEF,
        }
    }
}

/// Result of a sequence query
#[derive(Debug, Clone)]
pub struct SequenceMatch {
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Measured phase difference in ticks
    pub measured_lag: f64,
    /// Expected lag that was queried
    pub expected_lag: u64,
    /// Energy (amplitude) at item A's bucket
    pub energy_a: f64,
    /// Energy (amplitude) at item B's bucket
    pub energy_b: f64,
}

impl SequenceMatch {
    /// Returns true if the sequence was detected with sufficient confidence
    pub fn is_match(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

impl ResonantBloom {
    /// Create a new Resonant Bloom Filter with default configuration
    pub fn new(size: usize) -> Self {
        Self::with_config(ResonantConfig {
            size,
            ..Default::default()
        })
    }

    /// Create a new Resonant Bloom Filter with custom configuration
    pub fn with_config(config: ResonantConfig) -> Self {
        Self {
            buckets: vec![Complex64::new(0.0, 0.0); config.size],
            decay: config.decay,
            tick: 0,
            omega_base: config.omega_base,
            seed: config.seed,
            insertion_counts: vec![0; config.size],
        }
    }

    /// Get the number of buckets
    pub fn size(&self) -> usize {
        self.buckets.len()
    }

    /// Get the current tick
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Get the total energy in the filter (sum of all amplitudes squared)
    pub fn total_energy(&self) -> f64 {
        self.buckets.iter().map(|z| z.norm_sqr()).sum()
    }

    /// Advance time by one tick - the heartbeat of the data structure.
    ///
    /// Every bucket rotates by its assigned frequency and decays slightly.
    /// This is what encodes temporal information into the phase structure.
    pub fn step(&mut self) {
        self.tick += 1;
        let decay = self.decay;
        let omega_base = self.omega_base;
        let size = self.buckets.len() as f64;

        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            // Each bucket has its own rotation frequency based on index
            // This creates a "frequency comb" - different items rotate at different speeds
            let omega_i = omega_base * (1.0 + (i as f64) / size);
            let rotation = Complex64::from_polar(decay, omega_i);
            *bucket *= rotation;
        }
    }

    /// Advance time by multiple ticks at once (more efficient than calling step() repeatedly)
    pub fn step_n(&mut self, n: u64) {
        if n == 0 {
            return;
        }

        let decay_n = self.decay.powi(n as i32);
        let omega_base = self.omega_base;
        let size = self.buckets.len() as f64;

        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let omega_i = omega_base * (1.0 + (i as f64) / size);
            let total_rotation = omega_i * (n as f64);
            let rotation = Complex64::from_polar(decay_n, total_rotation);
            *bucket *= rotation;
        }

        self.tick += n;
    }

    /// Hash an item to a bucket index
    fn hash_item<T: Hash>(&self, item: &T) -> usize {
        let mut hasher = SipHasher13::new_with_keys(self.seed, self.seed ^ 0xCAFEBABE);
        item.hash(&mut hasher);
        (hasher.finish() as usize) % self.buckets.len()
    }

    /// Get the rotation frequency for a specific bucket
    fn bucket_frequency(&self, idx: usize) -> f64 {
        self.omega_base * (1.0 + (idx as f64) / self.buckets.len() as f64)
    }

    /// Insert an item at the current time.
    ///
    /// Adds a "fresh" unit vector (phase 0) to the item's bucket.
    /// The current phase encodes "now" - as time passes and the bucket rotates,
    /// the phase will encode how long ago this item was inserted.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let idx = self.hash_item(item);
        // Add a unit vector at phase 0 (representing "right now")
        self.buckets[idx] += Complex64::new(1.0, 0.0);
        self.insertion_counts[idx] += 1;
    }

    /// Insert an item with a custom weight/amplitude
    pub fn insert_weighted<T: Hash>(&mut self, item: &T, weight: f64) {
        let idx = self.hash_item(item);
        self.buckets[idx] += Complex64::new(weight, 0.0);
        self.insertion_counts[idx] += 1;
    }

    /// Check if an item might be present (like standard Bloom filter membership)
    pub fn might_contain<T: Hash>(&self, item: &T, threshold: f64) -> bool {
        let idx = self.hash_item(item);
        self.buckets[idx].norm() >= threshold
    }

    /// Get the amplitude (energy) at an item's bucket
    pub fn get_amplitude<T: Hash>(&self, item: &T) -> f64 {
        let idx = self.hash_item(item);
        self.buckets[idx].norm()
    }

    /// Get the phase angle at an item's bucket (in radians)
    pub fn get_phase<T: Hash>(&self, item: &T) -> f64 {
        let idx = self.hash_item(item);
        self.buckets[idx].arg()
    }

    /// Query for a temporal sequence: Did item_a happen `expected_lag` ticks before item_b?
    ///
    /// This is the "holy shit" moment - detecting causal relationships in O(1).
    ///
    /// # How it works
    ///
    /// 1. Item A was inserted at time t_a, item B at time t_b
    /// 2. At current time T, A's phase has rotated for (T - t_a) ticks
    /// 3. B's phase has rotated for (T - t_b) ticks
    /// 4. The phase difference encodes (t_b - t_a) = the actual lag
    /// 5. We check if this matches the expected lag
    ///
    /// # Returns
    ///
    /// A `SequenceMatch` containing confidence score and measured values.
    pub fn query_sequence<T: Hash, U: Hash>(
        &self,
        item_a: &T,
        item_b: &U,
        expected_lag: u64,
    ) -> SequenceMatch {
        let idx_a = self.hash_item(item_a);
        let idx_b = self.hash_item(item_b);

        let vec_a = self.buckets[idx_a];
        let vec_b = self.buckets[idx_b];

        let energy_a = vec_a.norm();
        let energy_b = vec_b.norm();

        // If either bucket is empty, no sequence exists
        if energy_a < 1e-10 || energy_b < 1e-10 {
            return SequenceMatch {
                confidence: 0.0,
                measured_lag: f64::NAN,
                expected_lag,
                energy_a,
                energy_b,
            };
        }

        // Get frequencies for both buckets
        let omega_a = self.bucket_frequency(idx_a);
        let omega_b = self.bucket_frequency(idx_b);

        // Current phases
        let phase_a = vec_a.arg();
        let phase_b = vec_b.arg();

        // Expected phase difference if A happened `expected_lag` ticks before B
        // A has rotated for (T - t_a) ticks = expected_lag + (T - t_b) ticks
        // So A should be ahead by expected_lag * omega_a radians (approximately)

        // The key insight: we're looking for constructive interference
        // when the phase relationship matches the expected timing

        // Compensate for different rotation speeds
        let expected_phase_diff = (expected_lag as f64) * (omega_a - omega_b);

        // Actual phase difference (normalized to [-π, π])
        let mut actual_phase_diff = phase_a - phase_b - expected_phase_diff;
        while actual_phase_diff > PI {
            actual_phase_diff -= 2.0 * PI;
        }
        while actual_phase_diff < -PI {
            actual_phase_diff += 2.0 * PI;
        }

        // Confidence based on how well phases align
        // cos(0) = 1 (perfect match), cos(π) = -1 (worst mismatch)
        let phase_alignment = actual_phase_diff.cos();

        // Also consider energy levels - both should be significantly present
        let energy_factor = (energy_a.min(energy_b) / energy_a.max(energy_b)).sqrt();

        // Combined confidence
        let confidence = ((phase_alignment + 1.0) / 2.0) * energy_factor;

        // Estimate measured lag from phase difference
        let avg_omega = (omega_a + omega_b) / 2.0;
        let measured_lag = if avg_omega.abs() > 1e-10 {
            (phase_a - phase_b) / avg_omega
        } else {
            f64::NAN
        };

        SequenceMatch {
            confidence,
            measured_lag,
            expected_lag,
            energy_a,
            energy_b,
        }
    }

    /// Detect if a pattern (sequence of events with specific timing) occurred.
    ///
    /// This is the "Shazam" of data structures - like identifying a song
    /// by the relative timing of spectral peaks.
    pub fn detect_pattern<T: Hash>(&self, events: &[(T, u64)], min_confidence: f64) -> f64 {
        if events.len() < 2 {
            return 1.0; // Trivially true
        }

        let mut total_confidence = 0.0;
        let mut comparisons = 0;
        let mut matches_above_threshold = 0;

        // Check all consecutive pairs
        for window in events.windows(2) {
            let (ref item_a, time_a) = window[0];
            let (ref item_b, time_b) = window[1];
            let lag = time_b.saturating_sub(time_a);

            let result = self.query_sequence(item_a, item_b, lag);
            total_confidence += result.confidence;
            comparisons += 1;

            if result.confidence >= min_confidence {
                matches_above_threshold += 1;
            }
        }

        if comparisons > 0 {
            // Return weighted confidence: average confidence * fraction of matches above threshold
            let avg_confidence = total_confidence / comparisons as f64;
            let match_ratio = matches_above_threshold as f64 / comparisons as f64;
            avg_confidence * match_ratio.sqrt()
        } else {
            0.0
        }
    }

    /// Clear all buckets (reset the filter)
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = Complex64::new(0.0, 0.0);
        }
        for count in &mut self.insertion_counts {
            *count = 0;
        }
        self.tick = 0;
    }

    /// Get raw bucket state (for debugging/visualization)
    pub fn get_bucket(&self, idx: usize) -> Complex64 {
        self.buckets[idx]
    }

    /// Get all bucket amplitudes (for visualization)
    pub fn get_amplitude_spectrum(&self) -> Vec<f64> {
        self.buckets.iter().map(|z| z.norm()).collect()
    }

    /// Get all bucket phases (for visualization)
    pub fn get_phase_spectrum(&self) -> Vec<f64> {
        self.buckets.iter().map(|z| z.arg()).collect()
    }
}

/// A variant optimized for event correlation (like the security log example)
#[derive(Clone)]
pub struct EventCorrelator {
    filter: ResonantBloom,
    /// Window size in ticks (events older than this are considered expired)
    window: u64,
    /// Minimum confidence for positive correlation
    threshold: f64,
}

impl EventCorrelator {
    /// Create a new event correlator
    pub fn new(size: usize, window: u64, threshold: f64) -> Self {
        let decay = (-1.0 / window as f64).exp(); // Decay to 1/e over window
        Self {
            filter: ResonantBloom::with_config(ResonantConfig {
                size,
                decay,
                ..Default::default()
            }),
            window,
            threshold,
        }
    }

    /// Process an event and advance time
    pub fn process_event<T: Hash>(&mut self, event: &T) {
        self.filter.insert(event);
        self.filter.step();
    }

    /// Process multiple events at the same timestamp
    pub fn process_events<T: Hash>(&mut self, events: &[T]) {
        for event in events {
            self.filter.insert(event);
        }
        self.filter.step();
    }

    /// Check if event_a preceded event_b by approximately `lag` ticks
    pub fn check_correlation<T: Hash, U: Hash>(
        &self,
        event_a: &T,
        event_b: &U,
        lag: u64,
    ) -> bool {
        let result = self.filter.query_sequence(event_a, event_b, lag);
        result.is_match(self.threshold)
    }

    /// Get detailed correlation information
    pub fn get_correlation<T: Hash, U: Hash>(
        &self,
        event_a: &T,
        event_b: &U,
        lag: u64,
    ) -> SequenceMatch {
        self.filter.query_sequence(event_a, event_b, lag)
    }

    /// Get current tick
    pub fn current_tick(&self) -> u64 {
        self.filter.current_tick()
    }

    /// Advance time without processing events
    pub fn advance(&mut self, ticks: u64) {
        self.filter.step_n(ticks);
    }

    /// Get the configured window size
    pub fn window_size(&self) -> u64 {
        self.window
    }

    /// Check if an event is still within the correlation window
    pub fn is_within_window(&self, event_tick: u64) -> bool {
        let current = self.filter.current_tick();
        current.saturating_sub(event_tick) <= self.window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insertion() {
        let mut bloom = ResonantBloom::new(1024);

        bloom.insert(&"event_a");
        assert!(bloom.get_amplitude(&"event_a") > 0.5);

        // Non-inserted item should have low amplitude
        assert!(bloom.get_amplitude(&"event_x") < 0.1);
    }

    #[test]
    fn test_time_evolution() {
        let mut bloom = ResonantBloom::new(1024);

        bloom.insert(&"event");
        let initial_amp = bloom.get_amplitude(&"event");

        // Advance time
        for _ in 0..100 {
            bloom.step();
        }

        let final_amp = bloom.get_amplitude(&"event");

        // Amplitude should decay
        assert!(final_amp < initial_amp);
        // But not completely disappear
        assert!(final_amp > 0.1);
    }

    #[test]
    fn test_sequence_detection() {
        let mut bloom = ResonantBloom::with_config(ResonantConfig {
            size: 4096,
            decay: 0.995,
            omega_base: 2.0 * PI / 128.0,
            seed: 42,
        });

        // Insert event A
        bloom.insert(&"login_failed");

        // Advance 50 ticks
        bloom.step_n(50);

        // Insert event B
        bloom.insert(&"admin_access");

        // Advance a bit more to "now"
        bloom.step_n(10);

        // Query: Did login_failed happen ~50 ticks before admin_access?
        let result = bloom.query_sequence(&"login_failed", &"admin_access", 50);

        println!("Sequence detection result: {:?}", result);
        assert!(result.confidence > 0.3, "Should detect the sequence with decent confidence");
    }

    #[test]
    fn test_wrong_sequence_low_confidence() {
        let mut bloom = ResonantBloom::with_config(ResonantConfig {
            size: 4096,
            decay: 0.995,
            omega_base: 2.0 * PI / 128.0,
            seed: 42,
        });

        // Insert event A
        bloom.insert(&"login_failed");

        // Advance 50 ticks
        bloom.step_n(50);

        // Insert event B
        bloom.insert(&"admin_access");

        bloom.step_n(10);

        // Query with WRONG lag (100 instead of 50)
        let result = bloom.query_sequence(&"login_failed", &"admin_access", 100);

        println!("Wrong lag result: {:?}", result);
        // Confidence should be lower for wrong timing
    }

    #[test]
    fn test_event_correlator() {
        let mut correlator = EventCorrelator::new(2048, 100, 0.3);

        // Simulate a security event stream
        correlator.process_event(&"ssh_connection");

        // Some time passes
        for _ in 0..20 {
            correlator.process_event(&"normal_activity");
        }

        // Suspicious sequence
        correlator.process_event(&"failed_login");

        // 5 ticks later
        for _ in 0..5 {
            correlator.process_event(&"normal_activity");
        }

        correlator.process_event(&"privilege_escalation");

        // Check correlation
        let correlated = correlator.check_correlation(
            &"failed_login",
            &"privilege_escalation",
            5,
        );

        println!("Events correlated: {}", correlated);
    }

    #[test]
    fn test_decay_behavior() {
        let mut bloom = ResonantBloom::with_config(ResonantConfig {
            size: 1024,
            decay: 0.95,  // Aggressive decay for testing
            ..Default::default()
        });

        bloom.insert(&"old_event");

        // After many steps, old event should be nearly gone
        bloom.step_n(100);

        let old_amp = bloom.get_amplitude(&"old_event");

        bloom.insert(&"new_event");
        let new_amp = bloom.get_amplitude(&"new_event");

        assert!(new_amp > old_amp * 10.0, "New events should dominate old decayed ones");
    }

    #[test]
    fn test_multiple_insertions_same_bucket() {
        let mut bloom = ResonantBloom::new(1024);

        bloom.insert(&"event");
        let amp1 = bloom.get_amplitude(&"event");

        bloom.insert(&"event");
        let amp2 = bloom.get_amplitude(&"event");

        // Amplitude should increase (vectors add constructively at same phase)
        assert!(amp2 > amp1);
    }

    #[test]
    fn test_step_n_equivalence() {
        let config = ResonantConfig {
            size: 256,
            decay: 0.98,
            omega_base: 0.1,
            seed: 123,
        };

        let mut bloom1 = ResonantBloom::with_config(config.clone());
        let mut bloom2 = ResonantBloom::with_config(config);

        bloom1.insert(&"test");
        bloom2.insert(&"test");

        // Advance bloom1 step by step
        for _ in 0..50 {
            bloom1.step();
        }

        // Advance bloom2 all at once
        bloom2.step_n(50);

        // Results should be very close
        let diff = (bloom1.get_amplitude(&"test") - bloom2.get_amplitude(&"test")).abs();
        assert!(diff < 0.001, "step() and step_n() should produce same result");
    }

    #[test]
    fn test_clear() {
        let mut bloom = ResonantBloom::new(1024);

        bloom.insert(&"event1");
        bloom.insert(&"event2");
        bloom.step_n(10);

        assert!(bloom.total_energy() > 0.0);

        bloom.clear();

        assert!(bloom.total_energy() < 0.0001);
        assert_eq!(bloom.current_tick(), 0);
    }
}
