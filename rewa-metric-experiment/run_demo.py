import os
import subprocess
import json
import sys
def run_command(cmd):
    # Replace default python3 with the current interpreter to ensure venv usage
    if cmd.startswith("python3 "):
        cmd = cmd.replace("python3 ", f"{sys.executable} ")
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 1. Train Baseline (Normalized)
    print("\n=== Training Baseline Normalized ===")
    run_command("python3 train.py --modality vision --variant normalized --epochs 1 --dim 128 --batch_size 128")
    
    # 2. Train REWA
    print("\n=== Training REWA ===")
    run_command("python3 train.py --modality vision --variant rewa --epochs 1 --dim 128 --batch_size 128")
    
    # 3. Evaluate Baseline
    print("\n=== Evaluating Baseline ===")
    run_command("python3 evaluate.py --modality vision --variant normalized --dim 128 --batch_size 128 --checkpoint checkpoints/vision_normalized_s42.pt --output results/vision_normalized.json")
    
    # 4. Evaluate REWA
    print("\n=== Evaluating REWA ===")
    run_command("python3 evaluate.py --modality vision --variant rewa --dim 128 --batch_size 128 --checkpoint checkpoints/vision_rewa_s42.pt --output results/vision_rewa.json")
    
    # 5. Compare
    with open('results/vision_normalized.json') as f:
        naive = json.load(f)
    with open('results/vision_rewa.json') as f:
        rewa = json.load(f)
        
    print("\n" + "="*50)
    print("RESULTS COMPARISON (Vision - CIFAR-100 - 1 Epoch)")
    print("="*50)
    print(f"Metric          | Baseline (Norm) | REWA (Euclidean)")
    print(f"----------------|-----------------|-----------------")
    print(f"Recall@1        | {naive.get('recall_1',0):.4f}          | {rewa.get('recall_1',0):.4f}")
    print(f"Recall@5        | {naive.get('recall_5',0):.4f}          | {rewa.get('recall_5',0):.4f}")
    print(f"Intrinsic Dim   | {naive.get('intrinsic_dim_95',0)}             | {rewa.get('intrinsic_dim_95',0)}")
    print(f"Norm CV         | {naive.get('norm_cv',0):.4f}          | {rewa.get('norm_cv',0):.4f}")
    
    if rewa.get('recall_1', 0) > naive.get('recall_1', 0):
        print("\n✅ REWA outperforms Baseline!")
    else:
        print("\n❌ REWA underperforms (Needs more training?)")

if __name__ == "__main__":
    main()
