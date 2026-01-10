import time
import os
import zstandard as zstd
import lz4.frame
from codec import compress, decompress, Archive

def estimate_energy_proxy(raw_len, dec_time, compressed_len):
    """Energy ~ C1 * bytes_touched + C2 * compute_time"""
    bytes_touched = raw_len + compressed_len
    energy = (bytes_touched * 1e-9) + (dec_time * 0.1)
    return energy

def simulate_cache_misses(raw_len, block_size):
    l2_size = 256 * 1024
    if block_size < l2_size:
        return (raw_len // block_size) * 0.01
    return (raw_len // block_size) * 0.5

def run_benchmark(name, data):
    print(f"\n--- Benchmarking: {name} ({len(data)} bytes) ---")
    
    # Tier-1
    start = time.time()
    block_size = 16384
    c_t1 = compress(data, block_size=block_size, hyper=False)
    enc_t1 = time.time() - start
    Archive.save(c_t1, "temp_t1.ac")
    size_t1 = os.path.getsize("temp_t1.ac")
    
    start = time.time()
    d_t1 = decompress(c_t1)
    dec_t1 = time.time() - start
    t1_match = (d_t1 == data)
    
    tp_t1 = len(data) / dec_t1 if dec_t1 > 0 else 0
    eff_t1 = tp_t1 * (len(data) / size_t1) if size_t1 > 0 else 0
    energy_t1 = estimate_energy_proxy(len(data), dec_t1, size_t1)
    misses_t1 = simulate_cache_misses(len(data), block_size)

    # Tier-2 Hyper
    start = time.time()
    c_t2 = compress(data, hyper=True)
    enc_t2 = time.time() - start
    Archive.save(c_t2, "temp_t2.ac")
    size_t2 = os.path.getsize("temp_t2.ac")
    
    start = time.time()
    d_t2 = decompress(c_t2)
    dec_t2 = time.time() - start
    t2_match = (d_t2 == data)
    
    tp_t2 = len(data) / dec_t2 if dec_t2 > 0 else 0
    eff_t2 = tp_t2 * (len(data) / size_t2) if size_t2 > 0 else 0
    energy_t2 = estimate_energy_proxy(len(data), dec_t2, size_t2)
    misses_t2 = simulate_cache_misses(len(data), len(data))

    print(f"Tier-1: Ratio={len(data)/size_t1:.2f}, TP={tp_t1/1e6:.2f}MB/s, Energy={energy_t1*1e6:.2f}μJ, CacheEff={100/(1+misses_t1):.1f}%")
    print(f"Hyper : Ratio={len(data)/size_t2:.2f}, TP={tp_t2/1e6:.2f}MB/s, Energy={energy_t2*1e6:.2f}μJ, CacheEff={100/(1+misses_t2):.1f}%")
    
    # Baseline Zstd
    start = time.time()
    z_comp = zstd.ZstdCompressor(level=3).compress(data)
    z_enc = time.time() - start
    start = time.time()
    zstd.ZstdDecompressor().decompress(z_comp)
    z_dec = time.time() - start
    z_tp = len(data) / z_dec if z_dec > 0 else 0
    z_eff = z_tp * (len(data) / len(z_comp))

    print(f"Zstd  : Ratio={len(data)/len(z_comp):.2f}, Enc={z_enc:.4f}s, TP={z_tp/1e6:.2f}MB/s, Eff={z_eff/1e6:.2f}M-pts")
    
    return {
        "name": name,
        "ratio_t1": len(data)/size_t1,
        "ratio_t2": len(data)/size_t2,
        "eff_t1": eff_t1,
        "eff_t2": eff_t2,
        "z_eff": z_eff,
        "correct": t1_match and t2_match
    }

if __name__ == "__main__":
    test_data = {
        "Text": b"The quick brown fox jumps over the lazy dog. " * 500,
        "Binary": b"\x00\xff\x00\xaa" * 2000,
        "Random": os.urandom(10000)
    }
    
    results = []
    for name, data in test_data.items():
        results.append(run_benchmark(name, data))
        
    print("\nSummary (Efficiency Metric):")
    for r in results:
        print(f"{r['name']:<10}: T1={r['eff_t1']/1e6:.2f}M, T2={r['eff_t2']/1e6:.2f}M, Zstd={r['z_eff']/1e6:.2f}M {'PASS' if r['correct'] else 'FAIL'}")
