import os
import subprocess
import json
import sys

def run_command(cmd):
    if cmd.startswith("python3 "):
        cmd = cmd.replace("python3 ", f"{sys.executable} ")
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    os.makedirs('results_full', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    epochs = 5
    
    # 1. Normalized (Baseline A)
    print(f"\n=== Training Normalized (Baseline A) - {epochs} Epochs ===")
    run_command(f"python3 train.py --modality vision --variant normalized --epochs {epochs} --dim 128 --batch_size 128 --seed 42")
    run_command(f"python3 evaluate.py --modality vision --variant normalized --dim 128 --batch_size 128 --checkpoint checkpoints/vision_normalized_s42.pt --output results_full/vision_normalized_5e.json")
    
    # 2. Unnormalized (Baseline B) - Pure Euclidean, no Radial Head shenanigans (just raw vector)
    print(f"\n=== Training Unnormalized (Baseline B) - {epochs} Epochs ===")
    run_command(f"python3 train.py --modality vision --variant unnormalized --epochs {epochs} --dim 128 --batch_size 128 --seed 42")
    run_command(f"python3 evaluate.py --modality vision --variant unnormalized --dim 128 --batch_size 128 --checkpoint checkpoints/vision_unnormalized_s42.pt --output results_full/vision_unnormalized_5e.json")
    
    # 3. REWA (Proposed) - Best Config (Tau=0.1, Reg=0.0)
    print(f"\n=== Training REWA (Proposed) - {epochs} Epochs ===")
    # Using Reg=0.0 based on tuning
    run_command(f"python3 train.py --modality vision --variant rewa --epochs {epochs} --dim 128 --batch_size 128 --tau 0.1 --reg 0.0 --seed 42")
    run_command(f"python3 evaluate.py --modality vision --variant rewa --dim 128 --batch_size 128 --checkpoint checkpoints/vision_rewa_s42.pt --output results_full/vision_rewa_5e.json")
    
    # Compare
    results = {}
    for var in ['normalized', 'unnormalized', 'rewa']:
        with open(f"results_full/vision_{var}_5e.json") as f:
            results[var] = json.load(f)
            
    print("\n" + "="*60)
    print(f"FULL COMPARISON RESULTS (CIFAR-100 - {epochs} Epochs)")
    print("="*60)
    print(f"{'Metric':<15} | {'Norm (A)':<12} | {'Unnorm (B)':<12} | {'REWA (Ours)':<12}")
    print("-" * 60)
    
    metrics = ['recall_1', 'recall_5', 'recall_10', 'intrinsic_dim_95', 'norm_cv', 'norm_mean']
    for m in metrics:
        v_a = results['normalized'].get(m, 0)
        v_b = results['unnormalized'].get(m, 0)
        v_c = results['rewa'].get(m, 0)
        
        # Format
        if isinstance(v_a, float):
            print(f"{m:<15} | {v_a:.4f}       | {v_b:.4f}       | {v_c:.4f}")
        else:
            print(f"{m:<15} | {v_a:<12} | {v_b:<12} | {v_c:<12}")

if __name__ == "__main__":
    main()
