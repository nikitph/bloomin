//! # Resonant Bloom Filter Demo
//!
//! Demonstrates the "killer app": O(1) event correlation for detecting
//! attack patterns in streaming data.
//!
//! The Problem: You have a firehose of server logs (1M/sec). You want to detect
//! "Login Failed" followed by "Admin Access" within ~50ms.
//!
//! Standard Way: Buffer logs, index by user, run windowed join query. Slow, memory-heavy.
//! Resonant Way: Detect causal links in O(1) without buffering the stream.

use resonant_bloom::{EventCorrelator, ResonantBloom, ResonantConfig};
use std::f64::consts::PI;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         RESONANT BLOOM FILTER - THE TIME CRYSTAL            ║");
    println!("║   Temporal Event Correlation in O(1) Space and Time         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_basic_concepts();
    println!("\n{}\n", "═".repeat(64));

    demo_attack_detection();
    println!("\n{}\n", "═".repeat(64));

    demo_phase_encoding();
    println!("\n{}\n", "═".repeat(64));

    demo_shazam_pattern_matching();
}

fn demo_basic_concepts() {
    println!("📖 DEMO 1: Basic Concepts - Phasors as Memory\n");

    let mut bloom = ResonantBloom::with_config(ResonantConfig {
        size: 256,
        decay: 0.98,
        omega_base: 2.0 * PI / 64.0,
        seed: 42,
    });

    println!("Inserting 'event_A' at t=0...");
    bloom.insert(&"event_A");

    let initial_amp = bloom.get_amplitude(&"event_A");
    let initial_phase = bloom.get_phase(&"event_A");
    println!("  Amplitude: {:.4}", initial_amp);
    println!("  Phase: {:.4} rad\n", initial_phase);

    println!("Advancing 30 time steps (bucket rotates and decays)...\n");
    bloom.step_n(30);

    let later_amp = bloom.get_amplitude(&"event_A");
    let later_phase = bloom.get_phase(&"event_A");
    println!("After 30 ticks:");
    println!("  Amplitude: {:.4} (decayed by {:.1}%)", later_amp, (1.0 - later_amp / initial_amp) * 100.0);
    println!("  Phase: {:.4} rad (rotated by {:.4} rad)", later_phase, later_phase - initial_phase);

    println!("\n💡 Key Insight: The PHASE encodes WHEN the event happened!");
    println!("   Old events have rotated further than recent ones.");
}

fn demo_attack_detection() {
    println!("🔒 DEMO 2: Security Log Attack Pattern Detection\n");

    println!("Scenario: Detect 'LoginFailed' → 'AdminAccess' within 50 ticks");
    println!("(In production, 1 tick = 1ms, so 50 ticks = 50ms)\n");

    let mut correlator = EventCorrelator::new(4096, 200, 0.3);

    println!("📊 Simulating event stream...\n");

    // Normal activity for a while
    println!("  [t=0-20] Normal SSH activity");
    for _ in 0..20 {
        correlator.process_event(&"ssh_packet");
    }

    // The attack sequence
    println!("  [t=20] ⚠️  LoginFailed event");
    correlator.process_event(&"LoginFailed");

    // Some noise
    for _ in 0..3 {
        correlator.process_event(&"ssh_packet");
    }

    // 50 ticks later - the suspicious access
    println!("  [t=20-70] Normal activity...");
    correlator.advance(47);  // Total 50 ticks after LoginFailed

    println!("  [t=70] ⚠️  AdminAccess event");
    correlator.process_event(&"AdminAccess");

    // Check for the attack pattern
    println!("\n🔍 Querying for attack pattern...\n");

    let result = correlator.get_correlation(&"LoginFailed", &"AdminAccess", 50);

    println!("Query: Did 'LoginFailed' happen ~50 ticks before 'AdminAccess'?");
    println!("  Confidence: {:.2}%", result.confidence * 100.0);
    println!("  Expected lag: {} ticks", result.expected_lag);
    println!("  Energy at 'LoginFailed': {:.4}", result.energy_a);
    println!("  Energy at 'AdminAccess': {:.4}", result.energy_b);

    if result.is_match(0.3) {
        println!("\n🚨 ALERT: Attack pattern detected!");
    } else {
        println!("\n✅ No attack pattern detected.");
    }

    // Test with wrong timing
    println!("\n--- Checking false positive resistance ---\n");

    let wrong_result = correlator.get_correlation(&"LoginFailed", &"AdminAccess", 200);
    println!("Query: Did 'LoginFailed' happen ~200 ticks before 'AdminAccess'?");
    println!("  Confidence: {:.2}% (should be lower)", wrong_result.confidence * 100.0);
}

fn demo_phase_encoding() {
    println!("🌊 DEMO 3: Phase Difference Encodes Time Intervals\n");

    let mut bloom = ResonantBloom::with_config(ResonantConfig {
        size: 1024,
        decay: 0.995,
        omega_base: 2.0 * PI / 100.0,
        seed: 12345,
    });

    // Insert events at different times
    println!("Inserting events at different times:\n");

    bloom.insert(&"event_t0");
    println!("  'event_t0' inserted at t=0");

    bloom.step_n(25);
    bloom.insert(&"event_t25");
    println!("  'event_t25' inserted at t=25");

    bloom.step_n(25);
    bloom.insert(&"event_t50");
    println!("  'event_t50' inserted at t=50");

    bloom.step_n(25);
    bloom.insert(&"event_t75");
    println!("  'event_t75' inserted at t=75");

    // Move to observation time
    bloom.step_n(25);
    println!("\nNow at t=100. Querying phase differences:\n");

    let phase_0 = bloom.get_phase(&"event_t0");
    let phase_25 = bloom.get_phase(&"event_t25");
    let phase_50 = bloom.get_phase(&"event_t50");
    let phase_75 = bloom.get_phase(&"event_t75");

    println!("  Phase of event_t0:  {:.4} rad (inserted 100 ticks ago)", phase_0);
    println!("  Phase of event_t25: {:.4} rad (inserted 75 ticks ago)", phase_25);
    println!("  Phase of event_t50: {:.4} rad (inserted 50 ticks ago)", phase_50);
    println!("  Phase of event_t75: {:.4} rad (inserted 25 ticks ago)", phase_75);

    println!("\n💡 Events inserted earlier have rotated more!");
    println!("   The phase difference encodes the time interval.");

    // Test sequence queries
    println!("\nSequence Queries:");

    let r1 = bloom.query_sequence(&"event_t0", &"event_t25", 25);
    let r2 = bloom.query_sequence(&"event_t25", &"event_t50", 25);
    let r3 = bloom.query_sequence(&"event_t0", &"event_t50", 50);

    println!("  t0 → t25 (lag=25): confidence {:.2}%", r1.confidence * 100.0);
    println!("  t25 → t50 (lag=25): confidence {:.2}%", r2.confidence * 100.0);
    println!("  t0 → t50 (lag=50): confidence {:.2}%", r3.confidence * 100.0);
}

fn demo_shazam_pattern_matching() {
    println!("🎵 DEMO 4: Shazam-Style Pattern Detection\n");

    println!("Like Shazam identifies songs by spectral peak timing,");
    println!("Resonant Bloom identifies event sequences by phase relationships.\n");

    let mut bloom = ResonantBloom::with_config(ResonantConfig {
        size: 2048,
        decay: 0.99,
        omega_base: 2.0 * PI / 128.0,
        seed: 9999,
    });

    // Define the "fingerprint" of an attack pattern
    let attack_pattern: Vec<(&str, u64)> = vec![
        ("port_scan", 0),
        ("ssh_bruteforce", 10),
        ("login_success", 20),
        ("privilege_escalation", 25),
        ("data_exfiltration", 40),
    ];

    println!("📝 Attack pattern fingerprint:");
    for (event, time) in &attack_pattern {
        println!("  t={:3}: {}", time, event);
    }

    println!("\n📥 Simulating attack in progress...\n");

    // Insert the attack sequence
    for (event, time) in &attack_pattern {
        bloom.insert(event);
        println!("  [t={}] Event: {}", bloom.current_tick(), event);
        if *time < 40 {
            let next_time = attack_pattern.iter()
                .find(|(_, t)| *t > *time)
                .map(|(_, t)| t - time)
                .unwrap_or(0);
            bloom.step_n(next_time);
        }
    }

    bloom.step_n(10);  // Advance a bit past the pattern

    println!("\n🔍 Pattern matching...\n");

    // Check the pattern
    let confidence = bloom.detect_pattern(&attack_pattern, 0.5);

    println!("Pattern match confidence: {:.1}%", confidence * 100.0);

    if confidence > 0.5 {
        println!("\n🚨 MULTI-STAGE ATTACK DETECTED!");
        println!("   Pattern: Port Scan → SSH Bruteforce → Login → Privilege Escalation → Exfiltration");
    }

    // Compare with wrong pattern
    println!("\n--- Testing with incorrect pattern ---\n");

    let wrong_pattern: Vec<(&str, u64)> = vec![
        ("port_scan", 0),
        ("ssh_bruteforce", 100),  // Wrong timing!
        ("login_success", 200),
    ];

    let wrong_confidence = bloom.detect_pattern(&wrong_pattern, 0.5);
    println!("Wrong pattern confidence: {:.1}% (should be lower)", wrong_confidence * 100.0);
}
