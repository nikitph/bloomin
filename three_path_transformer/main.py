import torch
import argparse
import os
from utils import set_seed, load_config
from model import ThreePathTransformer
from memory import ConceptMemory
from training import TrainingProtocol
from evaluation import Evaluator
from visualization import create_all_plots

def train_and_evaluate(config, mode='threepath'):
    print(f"\nTraining {mode.upper()} Model...")
    
    vocab_size = 50 # Arbitrary sufficient size for color concepts
    
    model = ThreePathTransformer(
        vocab_size=vocab_size,
        dim=config['embedding_dim'],
        n_layers=config['n_layers'],
        sharpen_iters=config['sharpen_iters_slow'],
        sharpen_power=config['sharpen_power']
    )
    
    memory = ConceptMemory()
    tp = TrainingProtocol(model, memory, config)
    
    use_sleep = (mode == 'threepath')
    history = []
    
    for epoch in range(config['epochs']):
        metrics = tp.train_epoch(epoch, use_sleep=use_sleep)
        print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}, Wake_H={metrics['wake_entropy']:.3f}, Sleep_H={metrics['sleep_entropy']:.3f}")
        history.append(metrics)
        
    evaluator = Evaluator(tp)
    eval_results = evaluator.evaluate_all()
    eval_results['history'] = history
    
    print(f"  Final Depth-2 Acc: {eval_results['hierarchy'][2]:.2f}")
    
    return eval_results

def main():
    set_seed(42)
    config = load_config()
    config['max_depth'] = 5 # FORCE DEPTH 5 for this experiment
    os.makedirs('./results/metrics', exist_ok=True)
    
    print("="*60)
    print("Running Three-Path Transformer POC")
    print("="*60)
    
    # 1. Train Baseline
    baseline_results = train_and_evaluate(config, mode='baseline')
    
    # 2. Train Three-Path
    threepath_results = train_and_evaluate(config, mode='threepath')
    
    # 3. Combine results
    all_results = {
        'baseline': baseline_results,
        'threepath': threepath_results
    }
    
    # 4. Visualize
    print("\nGenerating Visualizations...")
    create_all_plots(all_results)
    
    # 5. Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    b_acc = baseline_results['hierarchy'][2]
    t_acc = threepath_results['hierarchy'][2]
    print(f"Baseline Depth-2 Accuracy: {b_acc:.1%}")
    print(f"Three-Path Depth-2 Accuracy: {t_acc:.1%}")
    if b_acc > 0:
        print(f"Improvement: {t_acc/b_acc:.2f}x")
    else:
        print("Improvement: Infinite (Baseline failed)")
    print("="*60)

if __name__ == "__main__":
    main()
