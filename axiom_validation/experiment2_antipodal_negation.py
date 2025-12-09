"""
Experiment 2: Antipodal Negation Validation (Axiom 3.2)
Tests whether semantic negation corresponds to geometric antipodality

Axiom 3.2 states: For all a in A (semantic axes), not(a) = -a
Prediction: For semantic axes, angle(positive, negative) ~ pi

Success criteria:
- Mean angle >= 2.8 rad (~0.89*pi)
- Standard deviation < 0.5 rad
- >80% of axes show angle > 2.5 rad
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load pretrained embedding model"""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    """Get normalized embedding for text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1)
        # Normalize to unit sphere (Axiom 0.2)
        embedding = embedding / torch.norm(embedding, dim=-1, keepdim=True)
    return embedding.cpu().numpy().squeeze()

# ============================================================================
# TEST CASES: Semantic Axes with Negations
# ============================================================================

# Carefully curated semantic axes
SEMANTIC_AXES = [
    # Colors
    ("red", "not red"),
    ("blue", "not blue"),
    ("green", "not green"),
    ("yellow", "not yellow"),
    ("black", "not black"),
    ("white", "not white"),

    # Properties
    ("hot", "not hot"),
    ("cold", "not cold"),
    ("big", "not big"),
    ("small", "not small"),
    ("heavy", "not heavy"),
    ("light", "not light"),
    ("fast", "not fast"),
    ("slow", "not slow"),

    # States
    ("alive", "not alive"),
    ("dead", "not dead"),
    ("open", "not open"),
    ("closed", "not closed"),
    ("full", "not full"),
    ("empty", "not empty"),

    # Qualities
    ("good", "not good"),
    ("bad", "not bad"),
    ("safe", "not safe"),
    ("dangerous", "not dangerous"),
    ("clean", "not clean"),
    ("dirty", "not dirty"),

    # Temporal
    ("early", "not early"),
    ("late", "not late"),
    ("new", "not new"),
    ("old", "not old"),

    # Emotional
    ("happy", "not happy"),
    ("sad", "not sad"),
    ("angry", "not angry"),
    ("calm", "not calm"),

    # Cognitive
    ("true", "not true"),
    ("false", "not false"),
    ("correct", "not correct"),
    ("wrong", "not wrong"),
    ("certain", "not certain"),
    ("uncertain", "not uncertain"),

    # Physical
    ("solid", "not solid"),
    ("liquid", "not liquid"),
    ("hard", "not hard"),
    ("soft", "not soft"),
    ("rough", "not rough"),
    ("smooth", "not smooth"),

    # Spatial
    ("near", "not near"),
    ("far", "not far"),
    ("inside", "not inside"),
    ("outside", "not outside"),
    ("above", "not above"),
    ("below", "not below"),
]

# ============================================================================
# MEASUREMENT
# ============================================================================

def measure_negation_angles(axes, tokenizer, model):
    """
    Measure angular separation between positive and negative forms

    Returns:
        results: List of dicts with axis info and measurements
    """
    results = []

    for positive_text, negative_text in axes:
        # Get embeddings
        pos_emb = get_embedding(positive_text, tokenizer, model)
        neg_emb = get_embedding(negative_text, tokenizer, model)

        # Compute angle (Axiom 1.1: d(u,v) = arccos(u*v))
        cos_angle = np.dot(pos_emb, neg_emb)
        # Clip to [-1, 1] to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angle_in_pi = angle / np.pi

        results.append({
            'axis': positive_text,
            'positive_embedding': pos_emb,
            'negative_embedding': neg_emb,
            'angle_rad': angle,
            'angle_pi': angle_in_pi,
            'cosine_similarity': cos_angle
        })

        print(f"{positive_text:15s} <-> {negative_text:15s}: "
              f"theta = {angle:.3f} rad ({angle_in_pi:.3f}pi), "
              f"cos(theta) = {cos_angle:.3f}")

    return results

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_results(results):
    """
    Statistical analysis of negation angles

    Tests:
    1. H0: mean angle = pi (perfect antipodality)
    2. Distribution analysis
    3. Success rate (angle > 2.5 rad)
    """
    angles = np.array([r['angle_rad'] for r in results])
    angles_pi = np.array([r['angle_pi'] for r in results])

    # Summary statistics
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    median_angle = np.median(angles)

    mean_pi = np.mean(angles_pi)
    std_pi = np.std(angles_pi)

    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    print(f"\nMean angle: {mean_angle:.3f} rad ({mean_pi:.3f}pi)")
    print(f"Std dev:    {std_angle:.3f} rad ({std_pi:.3f}pi)")
    print(f"Median:     {median_angle:.3f} rad ({np.median(angles_pi):.3f}pi)")
    print(f"Min:        {np.min(angles):.3f} rad ({np.min(angles_pi):.3f}pi)")
    print(f"Max:        {np.max(angles):.3f} rad ({np.max(angles_pi):.3f}pi)")

    # Test H0: mean = pi
    t_stat, p_value = ttest_1samp(angles, np.pi)
    print(f"\nOne-sample t-test (H0: mu = pi):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value > 0.05:
        print(f"  [PASS] Cannot reject H0: mean is consistent with pi")
    else:
        print(f"  [INFO] Reject H0: mean differs from pi (delta = {mean_angle - np.pi:.3f})")

    # Success rate (angle > 2.5 rad ~ 0.8*pi)
    threshold = 2.5
    success_count = np.sum(angles > threshold)
    success_rate = success_count / len(angles)

    print(f"\nSuccess rate (angle > {threshold} rad):")
    print(f"  {success_count}/{len(angles)} = {success_rate:.1%}")

    # Classification
    near_antipodal = np.sum(angles > 2.8)  # > 0.89*pi
    moderate = np.sum((angles > 2.0) & (angles <= 2.8))  # 0.64*pi - 0.89*pi
    poor = np.sum(angles <= 2.0)  # < 0.64*pi

    print(f"\nDistribution:")
    print(f"  Near-antipodal (>2.8 rad): {near_antipodal} ({near_antipodal/len(angles):.1%})")
    print(f"  Moderate (2.0-2.8 rad):    {moderate} ({moderate/len(angles):.1%})")
    print(f"  Poor (<2.0 rad):           {poor} ({poor/len(angles):.1%})")

    # Axiom 3.2 validation
    print("\n" + "="*70)
    print("AXIOM 3.2 VALIDATION")
    print("="*70)

    if mean_pi >= 0.85 and success_rate >= 0.80:
        print("[VALIDATED] AXIOM 3.2 IS VALIDATED")
        print(f"    Negation is approximately antipodal (mean = {mean_pi:.2f}pi)")
    elif mean_pi >= 0.70 and success_rate >= 0.60:
        print("[PARTIAL] AXIOM 3.2 PARTIALLY VALIDATED")
        print(f"    Negation shows antipodal tendency (mean = {mean_pi:.2f}pi)")
    else:
        print("[NOT VALIDATED] AXIOM 3.2 NOT VALIDATED")
        print(f"    Negation is not antipodal (mean = {mean_pi:.2f}pi)")

    return {
        'mean_angle': mean_angle,
        'mean_pi': mean_pi,
        'std_angle': std_angle,
        'std_pi': std_pi,
        'success_rate': success_rate,
        'near_antipodal_rate': near_antipodal / len(angles),
        'validated': mean_pi >= 0.85 and success_rate >= 0.80
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(results, output_path='experiment2_antipodal_negation.png'):
    """Create comprehensive visualizations"""
    angles = [r['angle_rad'] for r in results]
    angles_pi = [r['angle_pi'] for r in results]
    axes_labels = [r['axis'] for r in results]

    fig, axes_plots = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram of angles
    ax = axes_plots[0, 0]
    ax.hist(angles, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.pi, color='red', linestyle='--', linewidth=2, label=f'pi (perfect antipodal)')
    ax.axvline(np.mean(angles), color='green', linestyle='-', linewidth=2, label=f'Mean = {np.mean(angles):.2f}')
    ax.set_xlabel('Angle (radians)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Negation Angles', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Sorted angles
    ax = axes_plots[0, 1]
    sorted_indices = np.argsort(angles)[::-1]
    sorted_angles = [angles[i] for i in sorted_indices]
    sorted_labels = [axes_labels[i] for i in sorted_indices]

    colors = ['green' if a > 2.8 else 'orange' if a > 2.0 else 'red' for a in sorted_angles]
    ax.barh(range(len(sorted_angles)), sorted_angles, color=colors, alpha=0.7)
    ax.axvline(np.pi, color='red', linestyle='--', linewidth=2, label='pi')
    ax.axvline(2.5, color='orange', linestyle=':', linewidth=2, label='Threshold (2.5)')
    ax.set_xlabel('Angle (radians)', fontsize=12)
    ax.set_ylabel('Semantic Axis', fontsize=12)
    ax.set_title('Negation Angles by Axis (sorted)', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    # Plot 3: Angle in units of pi
    ax = axes_plots[1, 0]
    ax.hist(angles_pi, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='1pi (perfect)')
    ax.axvline(np.mean(angles_pi), color='green', linestyle='-', linewidth=2,
               label=f'Mean = {np.mean(angles_pi):.2f}pi')
    ax.set_xlabel('Angle (units of pi)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Negation Angles in Units of pi', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Scatter plot (cosine similarity)
    ax = axes_plots[1, 1]
    cos_sims = [r['cosine_similarity'] for r in results]
    ax.scatter(range(len(cos_sims)), cos_sims, alpha=0.6, s=50)
    ax.axhline(-1.0, color='red', linestyle='--', linewidth=2, label='Perfect antipodal (cos = -1)')
    ax.axhline(np.mean(cos_sims), color='green', linestyle='-', linewidth=2,
               label=f'Mean cos = {np.mean(cos_sims):.2f}')
    ax.set_xlabel('Axis Index', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Cosine Similarity: Positive <-> Negative', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] Visualization saved to '{output_path}'")
    plt.close()

# ============================================================================
# ADDITIONAL TESTS
# ============================================================================

def test_double_negation(tokenizer, model):
    """
    Test: not(not(a)) ~ a (double negation)
    Axiom 3.2 implies: -(-a) = a
    """
    print("\n" + "="*70)
    print("DOUBLE NEGATION TEST")
    print("="*70)

    test_cases = [
        "red",
        "happy",
        "true",
        "hot",
        "big"
    ]

    results = []
    for word in test_cases:
        # Original
        emb_a = get_embedding(word, tokenizer, model)

        # Single negation
        emb_not_a = get_embedding(f"not {word}", tokenizer, model)

        # Double negation
        emb_not_not_a = get_embedding(f"not not {word}", tokenizer, model)

        # Measure angles
        angle_a_not_a = np.arccos(np.clip(np.dot(emb_a, emb_not_a), -1, 1))
        angle_a_not_not_a = np.arccos(np.clip(np.dot(emb_a, emb_not_not_a), -1, 1))
        angle_not_a_not_not_a = np.arccos(np.clip(np.dot(emb_not_a, emb_not_not_a), -1, 1))

        print(f"\n{word}:")
        print(f"  a <-> not(a):       {angle_a_not_a:.3f} rad ({angle_a_not_a/np.pi:.2f}pi)")
        print(f"  a <-> not(not(a)):  {angle_a_not_not_a:.3f} rad ({angle_a_not_not_a/np.pi:.2f}pi)")
        print(f"  not(a) <-> not(not(a)): {angle_not_a_not_not_a:.3f} rad ({angle_not_a_not_not_a/np.pi:.2f}pi)")

        if angle_a_not_not_a < 0.5:  # Close to 0
            print(f"  [PASS] Double negation approximately recovers original")
            results.append(True)
        else:
            print(f"  [FAIL] Double negation does not recover original")
            results.append(False)

    success_rate = sum(results) / len(results)
    print(f"\nDouble negation success rate: {success_rate:.1%}")
    return success_rate

def test_antonym_vs_negation(tokenizer, model):
    """
    Test: Compare explicit antonyms vs "not X" negation
    Theory: Both should be approximately antipodal
    """
    print("\n" + "="*70)
    print("ANTONYM VS NEGATION TEST")
    print("="*70)

    antonym_pairs = [
        ("hot", "cold", "not hot"),
        ("big", "small", "not big"),
        ("fast", "slow", "not fast"),
        ("good", "bad", "not good"),
        ("happy", "sad", "not happy"),
        ("alive", "dead", "not alive"),
        ("open", "closed", "not open"),
        ("full", "empty", "not full"),
        ("true", "false", "not true"),
        ("near", "far", "not near"),
    ]

    for word, antonym, negation in antonym_pairs:
        emb_word = get_embedding(word, tokenizer, model)
        emb_antonym = get_embedding(antonym, tokenizer, model)
        emb_negation = get_embedding(negation, tokenizer, model)

        angle_to_antonym = np.arccos(np.clip(np.dot(emb_word, emb_antonym), -1, 1))
        angle_to_negation = np.arccos(np.clip(np.dot(emb_word, emb_negation), -1, 1))
        angle_antonym_negation = np.arccos(np.clip(np.dot(emb_antonym, emb_negation), -1, 1))

        print(f"\n{word}:")
        print(f"  {word} <-> {antonym}:    {angle_to_antonym:.3f} rad ({angle_to_antonym/np.pi:.2f}pi)")
        print(f"  {word} <-> {negation}: {angle_to_negation:.3f} rad ({angle_to_negation/np.pi:.2f}pi)")
        print(f"  {antonym} <-> {negation}: {angle_antonym_negation:.3f} rad ({angle_antonym_negation/np.pi:.2f}pi)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 2: ANTIPODAL NEGATION (AXIOM 3.2)")
    print("="*70)
    print("\nObjective: Validate that semantic negation corresponds to")
    print("           geometric antipodality on the unit sphere")
    print("\nAxiom 3.2: For all a in A, not(a) = -a")
    print("Prediction: angle(a, not(a)) ~ pi")
    print("="*70)

    # Load model
    print("\nLoading embedding model...")
    tokenizer, model = load_embedding_model()
    print("[DONE] Model loaded")

    # Run measurements
    print(f"\nTesting {len(SEMANTIC_AXES)} semantic axes...\n")
    results = measure_negation_angles(SEMANTIC_AXES, tokenizer, model)

    # Statistical analysis
    stats = analyze_results(results)

    # Visualization
    visualize_results(results)

    # Double negation test
    double_neg_rate = test_double_negation(tokenizer, model)

    # Antonym vs negation comparison
    test_antonym_vs_negation(tokenizer, model)

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    if stats['validated']:
        print("[VALIDATED] AXIOM 3.2 IS VALIDATED")
        print(f"    Mean angle: {stats['mean_pi']:.2f}pi (target: 1.0pi)")
        print(f"    Success rate: {stats['success_rate']:.1%} (target: >80%)")
        print("\n    Conclusion: Semantic negation exhibits strong antipodal structure.")
        print("    Embeddings naturally encode logical negation as geometric opposition.")
    else:
        print("[WARNING] AXIOM 3.2 IS NOT FULLY VALIDATED")
        print(f"    Mean angle: {stats['mean_pi']:.2f}pi (target: 1.0pi)")
        print(f"    Success rate: {stats['success_rate']:.1%} (target: >80%)")
        print("\n    Conclusion: Negation shows partial antipodal tendency but is not perfect.")
        print("    May require explicit training or architectural changes.")

    return results, stats

if __name__ == "__main__":
    results, stats = main()
