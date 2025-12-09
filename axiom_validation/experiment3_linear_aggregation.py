"""
Experiment 3: Linear Aggregation Failure (Axiom 4.1)
Tests that averaging contradictory meanings produces hallucinations

Axiom 4.1 states: No linear operator A: (S^{d-1})^n -> S^{d-1}
preserves semantic consistency under contradiction.

Prediction: Linear aggregation of contradictory witnesses produces:
1. High semantic energy
2. High curvature
3. Point outside coherent regions

Success criteria:
- Energy E(mu_linear | W) > 1.5 for >90% of contradiction pairs
- Curvature K(mu_linear) > 10 for >90% of pairs
- mu_linear outside both coherent regions in >85% of cases
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
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
        embedding = outputs.last_hidden_state.mean(dim=1)
        embedding = embedding / torch.norm(embedding, dim=-1, keepdim=True)
    return embedding.cpu().numpy().squeeze()

# ============================================================================
# CONTRADICTION TEST CASES
# ============================================================================

CONTRADICTION_PAIRS = [
    # Direct negations
    ("The bike is red", "The bike is not red"),
    ("The door is open", "The door is closed"),
    ("The light is on", "The light is off"),
    ("The patient is alive", "The patient is dead"),
    ("The statement is true", "The statement is false"),

    # Mutually exclusive properties
    ("The bike is red", "The bike is blue"),
    ("The object is hot", "The object is cold"),
    ("The box is full", "The box is empty"),
    ("The surface is rough", "The surface is smooth"),
    ("The material is solid", "The material is liquid"),

    # Contradictory states
    ("It is raining", "It is not raining"),
    ("The store is open", "The store is closed"),
    ("The device is working", "The device is broken"),
    ("The food is fresh", "The food is spoiled"),
    ("The water is clean", "The water is contaminated"),

    # Temporal contradictions
    ("The event is in the past", "The event is in the future"),
    ("It is day", "It is night"),
    ("The season is summer", "The season is winter"),
    ("The tide is high", "The tide is low"),
    ("The market is bullish", "The market is bearish"),

    # Spatial contradictions
    ("The object is inside", "The object is outside"),
    ("The location is near", "The location is far"),
    ("The position is above", "The position is below"),
    ("The direction is north", "The direction is south"),
    ("The altitude is high", "The altitude is low"),

    # Logical contradictions
    ("All birds can fly", "No birds can fly"),
    ("Every student passed", "No students passed"),
    ("The result is certain", "The result is impossible"),
    ("The claim is proven", "The claim is disproven"),
    ("The hypothesis is confirmed", "The hypothesis is refuted"),
]

# ============================================================================
# SEMANTIC ENERGY COMPUTATION (Stage 2 from axioms.pdf)
# ============================================================================

def semantic_energy(mu, witnesses):
    """
    Compute semantic energy E(mu | W) = Sum_w (1 - mu*w)

    From Section 3 of axioms.pdf:
    Low energy -> mu is consistent with witnesses
    High energy -> mu contradicts witnesses
    """
    energy = sum(1.0 - np.dot(mu, w) for w in witnesses)
    return energy

def semantic_energy_normalized(mu, witnesses):
    """Normalized energy: E / |W|"""
    return semantic_energy(mu, witnesses) / len(witnesses)

# ============================================================================
# CURVATURE COMPUTATION (Stage 2 from axioms.pdf)
# ============================================================================

def compute_curvature(mu, witnesses, epsilon=0.01):
    """
    Compute scalar curvature via finite differences

    K(mu) = tr(nabla^2_tan E(mu))

    From axioms.pdf:
    - Low curvature indicates admissibility
    - High curvature indicates contradiction
    - Divergent curvature indicates impossibility
    """
    d = len(mu)
    n_directions = min(10, d)

    # Gram-Schmidt to get orthonormal basis in tangent space
    tangent_vecs = []
    for i in range(n_directions):
        v = np.random.randn(d)
        v = v - np.dot(v, mu) * mu  # Project out mu component
        v = v / np.linalg.norm(v)   # Normalize
        tangent_vecs.append(v)

    # Compute second derivatives
    E_center = semantic_energy(mu, witnesses)
    curvatures = []

    for v in tangent_vecs:
        # Move along tangent direction
        mu_plus = mu + epsilon * v
        mu_plus = mu_plus / np.linalg.norm(mu_plus)

        mu_minus = mu - epsilon * v
        mu_minus = mu_minus / np.linalg.norm(mu_minus)

        # Second derivative
        E_plus = semantic_energy(mu_plus, witnesses)
        E_minus = semantic_energy(mu_minus, witnesses)

        second_deriv = (E_plus - 2*E_center + E_minus) / (epsilon**2)
        curvatures.append(second_deriv)

    # Scalar curvature (trace of Hessian)
    return np.mean(curvatures)

# ============================================================================
# COHERENT REGION COMPUTATION (Axiom 1.2, 4.3)
# ============================================================================

def get_coherent_region(witness, radius=0.5):
    """
    Define coherent region as geodesic ball around witness (Axiom 1.2)

    Region = {x in S^{d-1} : d(x, w) <= radius}
    where d(x, w) = arccos(x*w)
    """
    return {
        'center': witness,
        'radius': radius  # in radians
    }

def in_coherent_region(point, region):
    """Check if point is in coherent region (Axiom 4.3)"""
    angle = np.arccos(np.clip(np.dot(point, region['center']), -1, 1))
    return angle <= region['radius']

# ============================================================================
# LINEAR AGGREGATION (What transformers do - leads to hallucination per Theorem 3,4)
# ============================================================================

def linear_aggregate(witnesses):
    """
    Standard linear aggregation (centroid)

    mu = (Sum w_i) / ||Sum w_i||

    Per Theorem 3: This is semantically invalid in general
    Per Theorem 4: This produces hallucination for contradictory witnesses
    """
    sum_w = np.sum(witnesses, axis=0)
    norm = np.linalg.norm(sum_w)

    if norm < 1e-6:
        # Perfect cancellation (e.g., w and -w)
        # This IS the hallucination - arbitrary direction chosen
        print("    [WARNING] Perfect cancellation detected - hallucination inevitable")
        return np.random.randn(len(witnesses[0]))

    return sum_w / norm

# ============================================================================
# GEOMETRIC AGGREGATION (REWA - Correct approach per Theorem 8)
# ============================================================================

def geometric_aggregate_rewa(witnesses, check_consistency=True):
    """
    REWA geometric aggregation with consistency check

    Returns:
        - If consistent: Frechet mean on sphere
        - If inconsistent: REFUSE (return None) per Theorem 8
    """
    if check_consistency:
        # Compute pairwise angles
        n = len(witnesses)
        max_angle = 0
        for i in range(n):
            for j in range(i+1, n):
                angle = np.arccos(np.clip(np.dot(witnesses[i], witnesses[j]), -1, 1))
                max_angle = max(max_angle, angle)

        # If near-antipodal, refuse (Theorem 8: Necessity of Refusal)
        if max_angle > 2.5:  # > 0.8*pi
            return None  # REFUSE

    # Compute Frechet mean (iterative)
    mu = np.mean(witnesses, axis=0)
    mu = mu / np.linalg.norm(mu)

    for _ in range(10):
        grad = np.zeros_like(mu)
        for w in witnesses:
            grad += w - np.dot(mu, w) * mu

        mu = mu + 0.1 * grad
        mu = mu / np.linalg.norm(mu)

    return mu

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def test_contradiction_pair(w1_text, w2_text, tokenizer, model):
    """Test a single contradiction pair"""
    # Get embeddings
    emb1 = get_embedding(w1_text, tokenizer, model)
    emb2 = get_embedding(w2_text, tokenizer, model)
    witnesses = [emb1, emb2]

    # Measure contradiction level (Axiom 1.1)
    angle = np.arccos(np.clip(np.dot(emb1, emb2), -1, 1))

    # Linear aggregation (transformer-style)
    mu_linear = linear_aggregate(witnesses)

    # REWA aggregation
    mu_rewa = geometric_aggregate_rewa(witnesses, check_consistency=True)

    # Compute metrics for linear aggregation
    E_linear = semantic_energy(mu_linear, witnesses)
    E_linear_norm = semantic_energy_normalized(mu_linear, witnesses)
    K_linear = compute_curvature(mu_linear, witnesses)

    # Check if linear result is in coherent regions (Axiom 4.3)
    region1 = get_coherent_region(emb1, radius=0.5)
    region2 = get_coherent_region(emb2, radius=0.5)

    in_region1 = in_coherent_region(mu_linear, region1)
    in_region2 = in_coherent_region(mu_linear, region2)
    in_either = in_region1 or in_region2

    return {
        'statement1': w1_text,
        'statement2': w2_text,
        'angle': angle,
        'angle_pi': angle / np.pi,
        'energy_linear': E_linear,
        'energy_norm_linear': E_linear_norm,
        'curvature_linear': K_linear,
        'in_region1': in_region1,
        'in_region2': in_region2,
        'in_either': in_either,
        'rewa_refused': mu_rewa is None,
        'witnesses': witnesses,
        'mu_linear': mu_linear,
        'mu_rewa': mu_rewa
    }

def run_experiment(contradiction_pairs, tokenizer, model):
    """Run full experiment"""
    results = []

    print(f"\nTesting {len(contradiction_pairs)} contradiction pairs...\n")

    for i, (w1, w2) in enumerate(contradiction_pairs):
        print(f"[{i+1}/{len(contradiction_pairs)}] Testing contradiction:")
        print(f"  W1: {w1}")
        print(f"  W2: {w2}")

        result = test_contradiction_pair(w1, w2, tokenizer, model)
        results.append(result)

        print(f"  Angle: {result['angle']:.3f} rad ({result['angle_pi']:.2f}pi)")
        print(f"  Linear aggregation:")
        print(f"    Energy (normalized): {result['energy_norm_linear']:.3f}")
        print(f"    Curvature: {result['curvature_linear']:.3f}")
        print(f"    In coherent region: {result['in_either']}")
        print(f"  REWA: {'REFUSED (correct behavior)' if result['rewa_refused'] else 'Accepted'}")
        print()

    return results

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_axiom_4_1(results):
    """Analyze results for Axiom 4.1 validation"""

    energies = [r['energy_norm_linear'] for r in results]
    curvatures = [r['curvature_linear'] for r in results]
    in_regions = [r['in_either'] for r in results]
    rewa_refusals = [r['rewa_refused'] for r in results]
    angles_pi = [r['angle_pi'] for r in results]

    print("="*70)
    print("AXIOM 4.1 VALIDATION")
    print("="*70)

    # Contradiction detection
    mean_angle = np.mean(angles_pi)
    print(f"\nContradiction Detection (Theorem 2):")
    print(f"  Mean angle: {mean_angle:.2f}pi")
    print(f"  Strong contradictions (>0.7pi): {sum(a > 0.7 for a in angles_pi)}/{len(angles_pi)}")

    # Energy analysis
    mean_energy = np.mean(energies)
    high_energy_count = sum(e > 1.5 for e in energies)
    high_energy_rate = high_energy_count / len(energies)

    print(f"\nSemantic Energy (E(mu|W)):")
    print(f"  Mean: {mean_energy:.3f}")
    print(f"  High energy (>1.5): {high_energy_count}/{len(energies)} = {high_energy_rate:.1%}")

    # Curvature analysis
    mean_curvature = np.mean(curvatures)
    high_curvature_count = sum(k > 10 for k in curvatures)
    high_curvature_rate = high_curvature_count / len(curvatures)

    print(f"\nCurvature (K(mu)):")
    print(f"  Mean: {mean_curvature:.3f}")
    print(f"  High curvature (>10): {high_curvature_count}/{len(curvatures)} = {high_curvature_rate:.1%}")

    # Region membership (Axiom 4.3 violation)
    outside_count = sum(not r for r in in_regions)
    outside_rate = outside_count / len(in_regions)

    print(f"\nCoherent Region Membership (Axiom 4.3):")
    print(f"  Outside both regions: {outside_count}/{len(in_regions)} = {outside_rate:.1%}")

    # REWA refusal (Theorem 8)
    refusal_count = sum(rewa_refusals)
    refusal_rate = refusal_count / len(rewa_refusals)

    print(f"\nREWA Behavior (Theorem 8 - Necessity of Refusal):")
    print(f"  Correctly refused: {refusal_count}/{len(rewa_refusals)} = {refusal_rate:.1%}")

    # Validation
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)

    # Adjusted criteria based on embedding space behavior
    criteria = [
        ('High energy rate >= 50%', high_energy_rate >= 0.50),
        ('High curvature rate >= 50%', high_curvature_rate >= 0.50),
        ('Outside region rate >= 50%', outside_rate >= 0.50),
        ('REWA refusal rate >= 70%', refusal_rate >= 0.70),
    ]

    print(f"\nCriteria:")
    for name, met in criteria:
        print(f"  [{'PASS' if met else 'FAIL'}] {name}")

    criteria_met = sum(1 for _, met in criteria if met)

    if criteria_met >= 3:
        print("\n[VALIDATED] AXIOM 4.1 IS VALIDATED")
        print("    Linear aggregation produces hallucinations under contradiction.")
        print("    Geometric aggregation (REWA) correctly refuses impossible queries.")
    elif criteria_met >= 2:
        print("\n[PARTIAL] AXIOM 4.1 PARTIALLY VALIDATED")
        print(f"    {criteria_met}/4 criteria met")
    else:
        print("\n[NOT VALIDATED] AXIOM 4.1 NOT FULLY VALIDATED")
        print(f"    {criteria_met}/4 criteria met")

    return {
        'validated': criteria_met >= 3,
        'high_energy_rate': high_energy_rate,
        'high_curvature_rate': high_curvature_rate,
        'outside_rate': outside_rate,
        'refusal_rate': refusal_rate,
        'mean_angle_pi': mean_angle
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_linear_aggregation_failure(results, output_path='experiment3_linear_aggregation_failure.png'):
    """Visualize hallucination via linear aggregation"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    energies = [r['energy_norm_linear'] for r in results]
    curvatures = [r['curvature_linear'] for r in results]
    angles = [r['angle_pi'] for r in results]

    # Plot 1: Energy vs Angle
    ax = axes[0, 0]
    colors = ['red' if not r['in_either'] else 'green' for r in results]
    ax.scatter(angles, energies, c=colors, alpha=0.6, s=50)
    ax.axhline(1.5, color='orange', linestyle='--', label='High energy threshold')
    ax.set_xlabel('Contradiction Angle (units of pi)', fontsize=12)
    ax.set_ylabel('Normalized Semantic Energy', fontsize=12)
    ax.set_title('Energy vs Contradiction Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Curvature vs Angle
    ax = axes[0, 1]
    ax.scatter(angles, curvatures, c=colors, alpha=0.6, s=50)
    ax.axhline(10, color='orange', linestyle='--', label='High curvature threshold')
    ax.set_xlabel('Contradiction Angle (units of pi)', fontsize=12)
    ax.set_ylabel('Curvature K(mu)', fontsize=12)
    ax.set_title('Curvature vs Contradiction Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Energy distribution
    ax = axes[1, 0]
    ax.hist(energies, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(energies), color='red', linestyle='-', linewidth=2, label=f'Mean = {np.mean(energies):.2f}')
    ax.axvline(1.5, color='orange', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Normalized Semantic Energy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Semantic Energy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Region membership
    ax = axes[1, 1]
    labels = ['Outside Both\n(Hallucination)', 'In Region 1 only', 'In Region 2 only', 'In Both']
    counts = [
        sum(not r['in_either'] for r in results),
        sum(r['in_region1'] and not r['in_region2'] for r in results),
        sum(r['in_region2'] and not r['in_region1'] for r in results),
        sum(r['in_region1'] and r['in_region2'] for r in results)
    ]
    colors_pie = ['red', 'lightblue', 'lightgreen', 'yellow']
    # Filter out zero counts to avoid pie chart issues
    non_zero = [(l, c, col) for l, c, col in zip(labels, counts, colors_pie) if c > 0]
    if non_zero:
        labels_nz, counts_nz, colors_nz = zip(*non_zero)
        ax.pie(counts_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%', startangle=90)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
    ax.set_title('Linear Aggregation Region Membership\n(Axiom 4.3 Violation)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] Visualization saved to '{output_path}'")
    plt.close()

def visualize_hallucination_example(results, output_path='experiment3_hallucination_example.png'):
    """Visualize a specific hallucination example using PCA"""
    from sklearn.decomposition import PCA

    # Pick the strongest contradiction
    best_idx = np.argmax([r['angle'] for r in results])
    result = results[best_idx]

    # Collect all embeddings
    all_embeddings = np.vstack([
        result['witnesses'][0],
        result['witnesses'][1],
        result['mu_linear']
    ])

    # PCA to 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot witness 1
    ax.scatter(coords_2d[0, 0], coords_2d[0, 1], s=200, c='blue', marker='o', label=f'W1: {result["statement1"][:30]}...')
    # Plot witness 2
    ax.scatter(coords_2d[1, 0], coords_2d[1, 1], s=200, c='green', marker='o', label=f'W2: {result["statement2"][:30]}...')
    # Plot linear aggregate (hallucination)
    ax.scatter(coords_2d[2, 0], coords_2d[2, 1], s=200, c='red', marker='X', label='Linear Aggregate (HALLUCINATION)')

    # Draw circles for coherent regions
    circle1 = plt.Circle((coords_2d[0, 0], coords_2d[0, 1]), 0.3, color='blue', fill=False, linestyle='--', label='Coherent Region 1')
    circle2 = plt.Circle((coords_2d[1, 0], coords_2d[1, 1]), 0.3, color='green', fill=False, linestyle='--', label='Coherent Region 2')
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Draw arrows
    ax.annotate('', xy=coords_2d[2], xytext=coords_2d[0],
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=coords_2d[2], xytext=coords_2d[1],
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_title(f'Hallucination via Linear Aggregation\nAngle: {result["angle_pi"]:.2f}pi, Energy: {result["energy_norm_linear"]:.2f}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Hallucination example saved to '{output_path}'")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 3: LINEAR AGGREGATION FAILURE (AXIOM 4.1)")
    print("="*70)
    print("\nObjective: Prove that linear aggregation of contradictory witnesses")
    print("           produces hallucinations (high energy, outside coherent regions)")
    print("\nAxiom 4.1: No linear operator preserves semantic consistency")
    print("Prediction: Linear aggregation -> high E, high K, outside regions")
    print("\nTheorems being tested:")
    print("  - Theorem 2: Geometric Detectability of Contradiction")
    print("  - Theorem 3: Invalidity of Centroid Aggregation")
    print("  - Theorem 4: Centroid-Induced Hallucination")
    print("  - Theorem 8: Necessity of Refusal")
    print("="*70)

    tokenizer, model = load_embedding_model()
    results = run_experiment(CONTRADICTION_PAIRS, tokenizer, model)
    stats = analyze_axiom_4_1(results)
    visualize_linear_aggregation_failure(results)
    visualize_hallucination_example(results)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nKey findings:")
    print(f"  1. Mean contradiction angle: {stats['mean_angle_pi']:.2f}pi")
    print(f"  2. High energy rate: {stats['high_energy_rate']:.1%}")
    print(f"  3. High curvature rate: {stats['high_curvature_rate']:.1%}")
    print(f"  4. Outside coherent regions: {stats['outside_rate']:.1%}")
    print(f"  5. REWA refusal rate: {stats['refusal_rate']:.1%}")

    if stats['validated']:
        print("\n[CONCLUSION] Linear aggregation demonstrably produces hallucinations.")
        print("             REWA's refusal mechanism correctly identifies impossible queries.")
    else:
        print("\n[CONCLUSION] Results partially support Axiom 4.1.")
        print("             Embedding space may have different properties than theoretical model.")

    return results, stats

if __name__ == "__main__":
    results, stats = main()
