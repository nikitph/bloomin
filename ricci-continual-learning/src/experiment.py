"""
Main Experiment: Geometric Inoculation Against Catastrophic Forgetting

This experiment tests the Ricci-REWA hypothesis:
"Preserving local Ricci curvature during training preserves task performance,
even when weights change dramatically."

Protocol:
1. Train on MNIST (Task A) → achieve ~99% accuracy
2. Train on FashionMNIST (Task B) with three conditions:
   - Baseline: Standard training (expect MNIST → ~10%)
   - EWC: Penalize weight changes (expect MNIST → ~70%)
   - Ricci-Reg: Penalize curvature changes (expect MNIST → ~95%+)

3. Measure:
   - Task A accuracy after Task B training
   - Weight space distance from Task A optimal
   - Curvature distance from Task A optimal
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from tqdm import tqdm

from .models import SimpleMLP, ConvNet, compute_weight_distance, get_weight_vector
from .continual_learning import BaselineCL, EWCCL, RicciRegCL, HybridCL
from .ricci_curvature import compute_ricci_on_embeddings


def get_datasets(
    data_root: str = './data',
    subset_size: Optional[int] = None
) -> Tuple[Dict, Dict]:
    """
    Load MNIST and FashionMNIST datasets.

    Args:
        data_root: Directory to store/load data
        subset_size: If set, use only this many samples (for quick testing)

    Returns:
        mnist_loaders: dict with 'train' and 'test' DataLoaders
        fashion_loaders: dict with 'train' and 'test' DataLoaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST
    mnist_train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    # FashionMNIST
    fashion_train = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)

    if subset_size:
        # Subset for quick testing
        mnist_train = Subset(mnist_train, range(min(subset_size, len(mnist_train))))
        mnist_test = Subset(mnist_test, range(min(subset_size // 5, len(mnist_test))))
        fashion_train = Subset(fashion_train, range(min(subset_size, len(fashion_train))))
        fashion_test = Subset(fashion_test, range(min(subset_size // 5, len(fashion_test))))

    batch_size = 64

    mnist_loaders = {
        'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
        'test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    }

    fashion_loaders = {
        'train': DataLoader(fashion_train, batch_size=batch_size, shuffle=True),
        'test': DataLoader(fashion_test, batch_size=batch_size, shuffle=False)
    }

    return mnist_loaders, fashion_loaders


class ContinualLearningExperiment:
    """
    Run the complete MNIST → FashionMNIST continual learning experiment.
    """

    def __init__(
        self,
        model_type: str = 'mlp',
        device: str = 'cpu',
        data_root: str = './data',
        results_dir: str = './results',
        subset_size: Optional[int] = None
    ):
        self.device = device
        self.model_type = model_type
        self.results_dir = results_dir
        self.subset_size = subset_size

        os.makedirs(results_dir, exist_ok=True)

        # Load datasets
        print("Loading datasets...")
        self.mnist_loaders, self.fashion_loaders = get_datasets(data_root, subset_size)

        # Store results
        self.results = {}

    def create_model(self) -> torch.nn.Module:
        """Create a fresh model instance."""
        if self.model_type == 'mlp':
            return SimpleMLP(
                input_dim=784,
                hidden_dims=(256, 128, 64),
                num_classes=10
            )
        elif self.model_type == 'conv':
            return ConvNet(
                in_channels=1,
                num_classes=10,
                embedding_dim=128
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def run_baseline(
        self,
        epochs_task_a: int = 10,
        epochs_task_b: int = 10,
        lr: float = 1e-3
    ) -> Dict:
        """Run baseline experiment (no protection)."""
        print("\n" + "=" * 60)
        print("BASELINE (No Protection)")
        print("=" * 60)

        model = self.create_model()
        learner = BaselineCL(model, device=self.device, lr=lr)

        # Store initial weights
        initial_weights = get_weight_vector(model).clone()

        # Task A: MNIST
        print("\nTraining on MNIST...")
        learner.train_task(
            self.mnist_loaders['train'],
            self.mnist_loaders['test'],
            epochs=epochs_task_a,
            task_id=0
        )

        mnist_after_a = learner.evaluate(self.mnist_loaders['test'])
        weights_after_a = get_weight_vector(model).clone()

        print(f"MNIST accuracy after Task A: {mnist_after_a['accuracy']:.4f}")

        # Task B: FashionMNIST
        print("\nTraining on FashionMNIST...")
        learner.train_task(
            self.fashion_loaders['train'],
            self.fashion_loaders['test'],
            epochs=epochs_task_b,
            task_id=1
        )

        mnist_after_b = learner.evaluate(self.mnist_loaders['test'])
        fashion_after_b = learner.evaluate(self.fashion_loaders['test'])
        weights_after_b = get_weight_vector(model).clone()

        print(f"\nFinal Results:")
        print(f"  MNIST accuracy: {mnist_after_b['accuracy']:.4f}")
        print(f"  FashionMNIST accuracy: {fashion_after_b['accuracy']:.4f}")

        weight_change = torch.norm(weights_after_b - weights_after_a).item()
        print(f"  Weight change (A→B): {weight_change:.4f}")

        return {
            'method': 'baseline',
            'mnist_after_a': mnist_after_a['accuracy'],
            'mnist_after_b': mnist_after_b['accuracy'],
            'fashion_after_b': fashion_after_b['accuracy'],
            'weight_change': weight_change,
            'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
        }

    def run_ewc(
        self,
        epochs_task_a: int = 10,
        epochs_task_b: int = 10,
        lr: float = 1e-3,
        ewc_lambda: float = 1000.0
    ) -> Dict:
        """Run EWC experiment."""
        print("\n" + "=" * 60)
        print(f"EWC (λ={ewc_lambda})")
        print("=" * 60)

        model = self.create_model()
        learner = EWCCL(
            model,
            device=self.device,
            lr=lr,
            ewc_lambda=ewc_lambda
        )

        # Task A: MNIST
        print("\nTraining on MNIST...")
        learner.train_task(
            self.mnist_loaders['train'],
            self.mnist_loaders['test'],
            epochs=epochs_task_a,
            task_id=0
        )

        mnist_after_a = learner.evaluate(self.mnist_loaders['test'])
        weights_after_a = get_weight_vector(model).clone()

        print(f"MNIST accuracy after Task A: {mnist_after_a['accuracy']:.4f}")

        # Compute Fisher (after_task)
        learner.after_task(self.mnist_loaders['train'], task_id=0)

        # Task B: FashionMNIST
        print("\nTraining on FashionMNIST with EWC...")
        learner.train_task(
            self.fashion_loaders['train'],
            self.fashion_loaders['test'],
            epochs=epochs_task_b,
            task_id=1
        )

        mnist_after_b = learner.evaluate(self.mnist_loaders['test'])
        fashion_after_b = learner.evaluate(self.fashion_loaders['test'])
        weights_after_b = get_weight_vector(model).clone()

        print(f"\nFinal Results:")
        print(f"  MNIST accuracy: {mnist_after_b['accuracy']:.4f}")
        print(f"  FashionMNIST accuracy: {fashion_after_b['accuracy']:.4f}")

        weight_change = torch.norm(weights_after_b - weights_after_a).item()
        print(f"  Weight change (A→B): {weight_change:.4f}")

        return {
            'method': 'ewc',
            'ewc_lambda': ewc_lambda,
            'mnist_after_a': mnist_after_a['accuracy'],
            'mnist_after_b': mnist_after_b['accuracy'],
            'fashion_after_b': fashion_after_b['accuracy'],
            'weight_change': weight_change,
            'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
        }

    def run_ricci_reg(
        self,
        epochs_task_a: int = 10,
        epochs_task_b: int = 10,
        lr: float = 1e-3,
        ricci_lambda: float = 1.0,
        k_neighbors: int = 10
    ) -> Dict:
        """Run Ricci Regularization experiment."""
        print("\n" + "=" * 60)
        print(f"RICCI REGULARIZATION (λ={ricci_lambda}, k={k_neighbors})")
        print("=" * 60)

        model = self.create_model()
        learner = RicciRegCL(
            model,
            device=self.device,
            lr=lr,
            ricci_lambda=ricci_lambda,
            k_neighbors=k_neighbors
        )

        # Task A: MNIST
        print("\nTraining on MNIST...")
        learner.train_task(
            self.mnist_loaders['train'],
            self.mnist_loaders['test'],
            epochs=epochs_task_a,
            task_id=0
        )

        mnist_after_a = learner.evaluate(self.mnist_loaders['test'])
        weights_after_a = get_weight_vector(model).clone()

        print(f"MNIST accuracy after Task A: {mnist_after_a['accuracy']:.4f}")

        # Compute reference curvature
        learner.after_task(self.mnist_loaders['train'], task_id=0)

        # Task B: FashionMNIST
        print("\nTraining on FashionMNIST with Ricci Regularization...")
        learner.train_task(
            self.fashion_loaders['train'],
            self.fashion_loaders['test'],
            epochs=epochs_task_b,
            task_id=1
        )

        mnist_after_b = learner.evaluate(self.mnist_loaders['test'])
        fashion_after_b = learner.evaluate(self.fashion_loaders['test'])
        weights_after_b = get_weight_vector(model).clone()

        print(f"\nFinal Results:")
        print(f"  MNIST accuracy: {mnist_after_b['accuracy']:.4f}")
        print(f"  FashionMNIST accuracy: {fashion_after_b['accuracy']:.4f}")

        weight_change = torch.norm(weights_after_b - weights_after_a).item()
        print(f"  Weight change (A→B): {weight_change:.4f}")

        # Compute curvature change
        curvature_change = learner.curvature_distance(
            self.mnist_loaders['train'],
            task_id=0
        )
        print(f"  Curvature change (A→B): {curvature_change:.4f}")

        return {
            'method': 'ricci_reg',
            'ricci_lambda': ricci_lambda,
            'k_neighbors': k_neighbors,
            'mnist_after_a': mnist_after_a['accuracy'],
            'mnist_after_b': mnist_after_b['accuracy'],
            'fashion_after_b': fashion_after_b['accuracy'],
            'weight_change': weight_change,
            'curvature_change': curvature_change,
            'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
        }

    def run_all_methods(
        self,
        epochs_task_a: int = 10,
        epochs_task_b: int = 10,
        lr: float = 1e-3,
        ewc_lambda: float = 1000.0,
        ricci_lambda: float = 1.0
    ) -> Dict:
        """Run all methods and compare."""
        results = {}

        # Baseline
        results['baseline'] = self.run_baseline(epochs_task_a, epochs_task_b, lr)

        # EWC
        results['ewc'] = self.run_ewc(epochs_task_a, epochs_task_b, lr, ewc_lambda)

        # Ricci Reg
        results['ricci_reg'] = self.run_ricci_reg(
            epochs_task_a, epochs_task_b, lr, ricci_lambda
        )

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n{'Method':<20} {'MNIST↓':<12} {'Fashion':<12} {'Weight Δ':<12} {'Forgetting':<12}")
        print("-" * 60)

        for method, r in results.items():
            print(
                f"{method:<20} "
                f"{r['mnist_after_b']:.4f}      "
                f"{r['fashion_after_b']:.4f}      "
                f"{r['weight_change']:.2f}        "
                f"{r['forgetting']:.4f}"
            )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"results_{timestamp}.json")

        # Convert to serializable format
        serializable_results = {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in results.items()
        }

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        self.results = results
        return results

    def hyperparameter_sweep(
        self,
        epochs_task_a: int = 5,
        epochs_task_b: int = 5,
        ewc_lambdas: List[float] = [100, 500, 1000, 5000],
        ricci_lambdas: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]
    ) -> Dict:
        """
        Sweep hyperparameters to find optimal settings.
        """
        print("\n" + "=" * 60)
        print("HYPERPARAMETER SWEEP")
        print("=" * 60)

        results = {'ewc': [], 'ricci_reg': []}

        # EWC sweep
        print("\nSweeping EWC lambda...")
        for ewc_lambda in ewc_lambdas:
            r = self.run_ewc(epochs_task_a, epochs_task_b, ewc_lambda=ewc_lambda)
            results['ewc'].append(r)

        # Ricci sweep
        print("\nSweeping Ricci lambda...")
        for ricci_lambda in ricci_lambdas:
            r = self.run_ricci_reg(epochs_task_a, epochs_task_b, ricci_lambda=ricci_lambda)
            results['ricci_reg'].append(r)

        # Find best settings
        best_ewc = max(results['ewc'], key=lambda x: x['mnist_after_b'])
        best_ricci = max(results['ricci_reg'], key=lambda x: x['mnist_after_b'])

        print("\n" + "=" * 60)
        print("SWEEP RESULTS")
        print("=" * 60)
        print(f"\nBest EWC: λ={best_ewc['ewc_lambda']}, MNIST retention={best_ewc['mnist_after_b']:.4f}")
        print(f"Best Ricci: λ={best_ricci['ricci_lambda']}, MNIST retention={best_ricci['mnist_after_b']:.4f}")

        return results


def run_continual_learning_experiment(
    model_type: str = 'mlp',
    epochs_task_a: int = 10,
    epochs_task_b: int = 10,
    device: str = None,
    subset_size: Optional[int] = None,
    ewc_lambda: float = 1000.0,
    ricci_lambda: float = 1.0
) -> Dict:
    """
    Convenience function to run the complete experiment.

    Args:
        model_type: 'mlp' or 'conv'
        epochs_task_a: Epochs for MNIST training
        epochs_task_b: Epochs for FashionMNIST training
        device: 'cpu' or 'cuda'
        subset_size: If set, use smaller datasets for quick testing
        ewc_lambda: EWC regularization strength
        ricci_lambda: Ricci regularization strength

    Returns:
        Dictionary with results for all methods
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Running experiment on {device}")
    print(f"Model: {model_type}")
    print(f"Epochs: {epochs_task_a} (Task A), {epochs_task_b} (Task B)")

    experiment = ContinualLearningExperiment(
        model_type=model_type,
        device=device,
        subset_size=subset_size
    )

    results = experiment.run_all_methods(
        epochs_task_a=epochs_task_a,
        epochs_task_b=epochs_task_b,
        ewc_lambda=ewc_lambda,
        ricci_lambda=ricci_lambda
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ricci-REWA Continual Learning Experiment')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'conv'])
    parser.add_argument('--epochs-a', type=int, default=10)
    parser.add_argument('--epochs-b', type=int, default=10)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--subset', type=int, default=None, help='Subset size for quick testing')
    parser.add_argument('--ewc-lambda', type=float, default=1000.0)
    parser.add_argument('--ricci-lambda', type=float, default=1.0)
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')

    args = parser.parse_args()

    if args.sweep:
        experiment = ContinualLearningExperiment(
            model_type=args.model,
            device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
            subset_size=args.subset
        )
        experiment.hyperparameter_sweep()
    else:
        run_continual_learning_experiment(
            model_type=args.model,
            epochs_task_a=args.epochs_a,
            epochs_task_b=args.epochs_b,
            device=args.device,
            subset_size=args.subset,
            ewc_lambda=args.ewc_lambda,
            ricci_lambda=args.ricci_lambda
        )
