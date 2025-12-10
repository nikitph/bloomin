#!/usr/bin/env python3
"""
Train Rewa-Space Projection Head

Trains the projection head to enforce antipodal behavior along policy axes,
then validates that negation pairs become properly antipodal.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.rewa_space import (
    RewaProjectionHead,
    generate_training_data,
    TrainingPair
)
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt


def evaluate_before_after(rewa_head: RewaProjectionHead, test_pairs: list):
    """Compare base embeddings vs Rewa-space embeddings."""
    base_model = rewa_head.base_model

    print("\n" + "="*70)
    print("BEFORE vs AFTER COMPARISON")
    print("="*70)
    print(f"{'Pair':<50} {'Base°':<10} {'Rewa°':<10} {'Δ':<10}")
    print("-"*70)

    base_angles = []
    rewa_angles = []

    for pos, neg in test_pairs:
        # Base embedding angles
        base_pos = base_model.encode(pos, convert_to_numpy=True)
        base_neg = base_model.encode(neg, convert_to_numpy=True)
        base_pos = base_pos / np.linalg.norm(base_pos)
        base_neg = base_neg / np.linalg.norm(base_neg)
        base_dot = np.dot(base_pos, base_neg)
        base_angle = np.degrees(np.arccos(np.clip(base_dot, -1, 1)))

        # Rewa-space angles
        rewa_pos = rewa_head.project(pos)
        rewa_neg = rewa_head.project(neg)
        rewa_dot = np.dot(rewa_pos, rewa_neg)
        rewa_angle = np.degrees(np.arccos(np.clip(rewa_dot, -1, 1)))

        base_angles.append(base_angle)
        rewa_angles.append(rewa_angle)

        delta = rewa_angle - base_angle
        label = f"{pos[:22]}... / {neg[:22]}..."
        print(f"{label:<50} {base_angle:>6.1f}°    {rewa_angle:>6.1f}°    {delta:>+6.1f}°")

    print("-"*70)
    print(f"{'MEAN':<50} {np.mean(base_angles):>6.1f}°    {np.mean(rewa_angles):>6.1f}°    {np.mean(rewa_angles)-np.mean(base_angles):>+6.1f}°")
    print(f"{'TARGET (antipodal)':<50} {'180.0°':>10}")

    return base_angles, rewa_angles


def plot_results(base_angles, rewa_angles, test_pairs, save_path="rewa_space_training.png"):
    """Visualize the transformation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Before vs After scatter
    ax1 = axes[0, 0]
    x = range(len(test_pairs))
    ax1.scatter(x, base_angles, color='red', label='Base Embedding', s=100, alpha=0.7)
    ax1.scatter(x, rewa_angles, color='green', label='Rewa-Space', s=100, alpha=0.7)
    ax1.axhline(y=180, color='black', linestyle='--', label='Antipodal (180°)')
    ax1.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='Orthogonal (90°)')
    ax1.set_xlabel('Test Pair Index')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Negation Pair Angles: Before vs After Transformation')
    ax1.legend()
    ax1.set_ylim(0, 200)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(base_angles, bins=15, alpha=0.5, color='red', label='Base Embedding')
    ax2.hist(rewa_angles, bins=15, alpha=0.5, color='green', label='Rewa-Space')
    ax2.axvline(x=180, color='black', linestyle='--', label='Antipodal')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Angles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Improvement bar chart
    ax3 = axes[1, 0]
    improvements = np.array(rewa_angles) - np.array(base_angles)
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax3.bar(x, improvements, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xlabel('Test Pair Index')
    ax3.set_ylabel('Angle Change (degrees)')
    ax3.set_title('Improvement per Pair (Green = More Antipodal)')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    categories = ['Mean Angle\n(Before)', 'Mean Angle\n(After)', 'Target\n(Antipodal)']
    values = [np.mean(base_angles), np.mean(rewa_angles), 180]
    colors = ['red', 'green', 'blue']
    bars = ax4.bar(categories, values, color=colors, alpha=0.7)

    # Add value labels
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}°', ha='center', fontsize=12, fontweight='bold')

    ax4.set_ylabel('Angle (degrees)')
    ax4.set_title('Summary: Antipodal Transformation Effect')
    ax4.set_ylim(0, 200)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVED] Visualization saved to {save_path}")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║                    REWA-SPACE TRAINING                                ║
    ║                                                                       ║
    ║    Training projection head to enforce antipodal negation             ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Test pairs for evaluation
    test_pairs = [
        ("The sky is blue", "The sky is not blue"),
        ("I love this product", "I hate this product"),
        ("The answer is yes", "The answer is no"),
        ("The patient is alive", "The patient is dead"),
        ("The test passed", "The test failed"),
        ("Access is granted", "Access is denied"),
        ("The claim is valid", "The claim is invalid"),
        ("Application approved", "Application rejected"),
        ("Safe for use", "Dangerous and harmful"),
        ("Fully compliant", "Violates regulations"),
    ]

    # Initialize
    print("Initializing Rewa-Space projection head...")
    rewa_head = RewaProjectionHead(output_dim=128)

    # Evaluate BEFORE training
    print("\n" + "="*70)
    print("EVALUATION BEFORE TRAINING")
    print("="*70)

    before_eval = rewa_head.evaluate_antipodal(test_pairs)
    print(f"\nBefore training:")
    print(f"  Mean angle: {before_eval['mean_angle_deg']:.1f}°")
    print(f"  Antipodal success rate: {before_eval['antipodal_success_rate']:.1%}")

    # Generate training data
    print("\n" + "="*70)
    print("GENERATING TRAINING DATA")
    print("="*70)

    training_pairs = generate_training_data()
    print(f"Generated {len(training_pairs)} training pairs")

    # Count by axis
    axes = {}
    for pair in training_pairs:
        axes[pair.axis_name] = axes.get(pair.axis_name, 0) + 1
    print("Training pairs by axis:")
    for axis, count in axes.items():
        print(f"  {axis}: {count}")

    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    history = rewa_head.train(
        training_pairs=training_pairs,
        epochs=50,
        learning_rate=0.005,
        verbose=True
    )

    # Evaluate AFTER training
    print("\n" + "="*70)
    print("EVALUATION AFTER TRAINING")
    print("="*70)

    after_eval = rewa_head.evaluate_antipodal(test_pairs)
    print(f"\nAfter training:")
    print(f"  Mean angle: {after_eval['mean_angle_deg']:.1f}°")
    print(f"  Antipodal success rate: {after_eval['antipodal_success_rate']:.1%}")

    # Detailed comparison
    base_angles, rewa_angles = evaluate_before_after(rewa_head, test_pairs)

    # Plot results
    plot_results(base_angles, rewa_angles, test_pairs)

    # Save trained model
    model_path = "rewa_space_head.json"
    rewa_head.save(model_path)
    print(f"\n[SAVED] Trained model saved to {model_path}")

    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)

    improvement = after_eval['mean_angle_deg'] - before_eval['mean_angle_deg']
    print(f"Mean angle improvement: {before_eval['mean_angle_deg']:.1f}° → {after_eval['mean_angle_deg']:.1f}° ({improvement:+.1f}°)")
    print(f"Antipodal success rate: {before_eval['antipodal_success_rate']:.1%} → {after_eval['antipodal_success_rate']:.1%}")

    if after_eval['mean_angle_deg'] > 120:
        print("\n✓ SUCCESS: Rewa-space achieves significant antipodal separation!")
    else:
        print("\n⚠ Training improved angles but more epochs may be needed.")

    return rewa_head, history


if __name__ == "__main__":
    main()
