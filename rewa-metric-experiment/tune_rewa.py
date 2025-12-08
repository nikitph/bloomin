import os
import subprocess
import json
import itertools
import sys

def run_tuning():
    python_cmd = sys.executable
    
    # Grid
    # Focused range based on hypothesis (Euclidean needs higher Tau or lower Reg)
    taus = [0.1, 1.0, 2.0]
    lrs = [1e-3]
    regs = [0.0, 0.1]
    
    configs = list(itertools.product(taus, lrs, regs))
    results = []
    
    print(f"Running {len(configs)} configurations for REWA...")
    
    os.makedirs('tuning_results', exist_ok=True)
    
    best_recall = 0.0
    best_config = None
    
    for tau, lr, reg in configs:
        run_id = f"rewa_t{tau}_l{lr}_r{reg}"
        print(f"\n--- Tuning: {run_id} ---")
        
        # Train
        cmd_train = f"{python_cmd} train.py --modality vision --variant rewa --epochs 1 --dim 128 --batch_size 128 --tau {tau} --lr {lr} --reg {reg} --seed 42"
        subprocess.check_call(cmd_train, shell=True)
        
        # Rename checkpoint to avoid overwrite
        ckpt_src = "checkpoints/vision_rewa_s42.pt"
        ckpt_dst = f"checkpoints/{run_id}.pt"
        os.rename(ckpt_src, ckpt_dst)
        
        # Evaluate
        res_file = f"tuning_results/{run_id}.json"
        cmd_eval = f"{python_cmd} evaluate.py --modality vision --variant rewa --dim 128 --batch_size 128 --checkpoint {ckpt_dst} --output {res_file}"
        subprocess.check_call(cmd_eval, shell=True)
        
        # Read Metric
        with open(res_file) as f:
            data = json.load(f)
            r1 = data.get('recall_1', 0.0)
            
        results.append({
            'tau': tau, 'lr': lr, 'reg': reg, 'recall_1': r1
        })
        
        print(f"Result: R@1 = {r1:.4f}")
        
        if r1 > best_recall:
            best_recall = r1
            best_config = (tau, lr, reg)

    print("\n" + "="*50)
    print("TUNING LEADERBOARD (Baseline ~ 0.1879)")
    print("="*50)
    print("Tau   | LR     | Reg   | Recall@1")
    print("------|--------|-------|---------")
    # Sort by Recall desc
    results.sort(key=lambda x: x['recall_1'], reverse=True)
    
    for r in results:
        print(f"{r['tau']:<5} | {r['lr']:<6} | {r['reg']:<5} | {r['recall_1']:.4f}")
        
    print(f"\nBest Config: Tau={best_config[0]}, LR={best_config[1]}, Reg={best_config[2]}")
    if best_recall > 0.1879:
        print("✅ Found config BEATING Baseline!")
    else:
        print("❌ Still below Baseline.")

if __name__ == "__main__":
    run_tuning()
