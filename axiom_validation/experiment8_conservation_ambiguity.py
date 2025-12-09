"""
Experiment 8: Conservation of Ambiguity Validation

Tests the Conservation of Ambiguity principle from "Axioms of Semantic Geometry":
- Ambiguity can ONLY decrease via:
  1. Adding new witnesses (more constraints)
  2. Explicit bias injection
- Without these, ambiguity is conserved (cannot spontaneously decrease)

Key insight: This explains why AI can seem to "hallucinate" - it's not adding information,
it's selecting from an ambiguous space without proper bias disclosure.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')


class AmbiguityCalculator:
    """Computes semantic ambiguity from witness embeddings"""

    def __init__(self, model):
        self.model = model

    def compute_ambiguity(self, witnesses: list) -> dict:
        """
        Compute ambiguity as the "size" of the consistent region.

        For a witness set W, the consistent region is all meanings μ that satisfy
        μ · w > 0 for all w ∈ W (hemisphere constraint).

        We approximate this using:
        1. Solid angle of the consistent region
        2. Variance of possible directions
        3. Entropy-based measure
        """
        if len(witnesses) == 0:
            return {
                'ambiguity': 1.0,  # Maximum ambiguity
                'solid_angle': 4 * np.pi,  # Full sphere
                'variance': float('inf'),
                'num_witnesses': 0
            }

        # Get embeddings for witnesses
        embeddings = self.model.encode(witnesses, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute ambiguity measures

        # 1. Mean direction and spread
        mean_direction = np.mean(embeddings, axis=0)
        mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-10)

        # 2. Angular spread (variance of angles from mean)
        angles_from_mean = np.arccos(np.clip(embeddings @ mean_direction, -1, 1))
        angular_variance = np.var(angles_from_mean)

        # 3. Pairwise consistency (how much witnesses agree)
        if len(witnesses) > 1:
            pairwise_sims = embeddings @ embeddings.T
            # Remove diagonal
            mask = ~np.eye(pairwise_sims.shape[0], dtype=bool)
            pairwise_sims_off = pairwise_sims[mask]
            consistency = np.mean(pairwise_sims_off)
        else:
            consistency = 1.0

        # 4. Approximate solid angle of consistent region
        # For orthogonal constraints, each halves the sphere
        # Solid angle ≈ 4π / 2^n for n orthogonal witnesses
        # For non-orthogonal, we estimate based on pairwise angles
        if len(witnesses) > 1:
            # Compute effective dimensionality reduction
            pairwise_angles = np.arccos(np.clip(embeddings @ embeddings.T, -1, 1))
            np.fill_diagonal(pairwise_angles, 0)
            avg_angle = np.mean(pairwise_angles[np.triu_indices(len(witnesses), k=1)])

            # Orthogonality factor (1 = orthogonal, 0 = parallel)
            orthogonality = avg_angle / (np.pi / 2)
            effective_constraints = len(witnesses) * min(orthogonality, 1.0)

            # Approximate solid angle reduction
            solid_angle = 4 * np.pi / (2 ** effective_constraints)
        else:
            solid_angle = 2 * np.pi  # Single witness halves the sphere

        # Normalized ambiguity (0 = no ambiguity, 1 = maximum)
        # Based on solid angle relative to full sphere
        ambiguity = solid_angle / (4 * np.pi)

        return {
            'ambiguity': float(ambiguity),
            'solid_angle': float(solid_angle),
            'angular_variance': float(angular_variance),
            'consistency': float(consistency),
            'num_witnesses': len(witnesses),
            'mean_direction': mean_direction,
            'embeddings': embeddings
        }

    def compute_spherical_convex_hull(self, embeddings: np.ndarray) -> dict:
        """
        Compute the spherical convex hull of witness directions.
        This represents the "boundary" of the consistent region.
        """
        if len(embeddings) < 4:
            return {
                'has_hull': False,
                'vertices': len(embeddings),
                'volume': None
            }

        try:
            # Project to lower dimension for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            projected = pca.fit_transform(embeddings)
            projected = projected / np.linalg.norm(projected, axis=1, keepdims=True)

            hull = ConvexHull(projected)
            return {
                'has_hull': True,
                'vertices': len(hull.vertices),
                'volume': hull.volume,
                'area': hull.area,
                'projected_points': projected
            }
        except Exception as e:
            return {
                'has_hull': False,
                'vertices': len(embeddings),
                'volume': None,
                'error': str(e)
            }

    def satisfies_hemisphere_constraint(self, meaning: np.ndarray, witnesses: list) -> dict:
        """
        Check if a meaning μ satisfies the hemisphere constraint for all witnesses.
        μ · w > 0 for all w ∈ W
        """
        if len(witnesses) == 0:
            return {
                'satisfies': True,
                'violations': [],
                'min_dot_product': float('inf')
            }

        embeddings = self.model.encode(witnesses, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        meaning = meaning / (np.linalg.norm(meaning) + 1e-10)

        dot_products = embeddings @ meaning
        violations = [witnesses[i] for i, dp in enumerate(dot_products) if dp <= 0]

        return {
            'satisfies': len(violations) == 0,
            'violations': violations,
            'num_violations': len(violations),
            'min_dot_product': float(np.min(dot_products)),
            'dot_products': dot_products.tolist()
        }


class RefusalPolicy:
    """Determines when to refuse based on ambiguity threshold"""

    def __init__(self, ambiguity_calculator: AmbiguityCalculator, threshold: float = 0.5):
        self.calculator = ambiguity_calculator
        self.threshold = threshold

    def should_refuse(self, witnesses: list) -> dict:
        """
        Refuse if ambiguity exceeds threshold and bias is not disclosed.
        """
        amb_result = self.calculator.compute_ambiguity(witnesses)
        ambiguity = amb_result['ambiguity']

        should_refuse = ambiguity > self.threshold

        return {
            'should_refuse': should_refuse,
            'ambiguity': ambiguity,
            'threshold': self.threshold,
            'reason': f"Ambiguity {ambiguity:.3f} {'>' if should_refuse else '<='} threshold {self.threshold}"
        }

    def generate_refusal_explanation(self, witnesses: list) -> str:
        """Generate explanation for why a response requires more information"""
        amb_result = self.calculator.compute_ambiguity(witnesses)

        if amb_result['num_witnesses'] == 0:
            return "I need more context to provide a meaningful response. What specific aspect are you interested in?"
        elif amb_result['num_witnesses'] == 1:
            return f"The query '{witnesses[0]}' is ambiguous. Could you provide additional context or constraints?"
        else:
            return f"Given the constraints {witnesses}, there are still multiple valid interpretations. The ambiguity level is {amb_result['ambiguity']:.1%}. Please specify which aspect you'd like me to focus on."


class BiasTracker:
    """Tracks and discloses biases applied to reduce ambiguity"""

    def __init__(self, ambiguity_calculator: AmbiguityCalculator):
        self.calculator = ambiguity_calculator
        self.biases = []

    def register_bias(self, bias_name: str, bias_direction: np.ndarray, reason: str = ""):
        """Register a bias that will reduce ambiguity"""
        self.biases.append({
            'name': bias_name,
            'direction': bias_direction / (np.linalg.norm(bias_direction) + 1e-10),
            'reason': reason
        })

    def apply_bias(self, witnesses: list, bias_name: str) -> dict:
        """
        Apply a registered bias and measure ambiguity reduction.
        Bias acts as an additional "virtual witness" that constrains the space.
        """
        # Get baseline ambiguity
        baseline = self.calculator.compute_ambiguity(witnesses)

        # Find the bias
        bias = None
        for b in self.biases:
            if b['name'] == bias_name:
                bias = b
                break

        if bias is None:
            return {
                'success': False,
                'error': f"Bias '{bias_name}' not registered"
            }

        # Create a "virtual witness" from the bias direction
        # We'll encode a dummy sentence and replace with bias direction
        virtual_witness = f"[BIAS: {bias_name}]"
        witnesses_with_bias = witnesses + [virtual_witness]

        # Get embeddings for original witnesses
        if len(witnesses) > 0:
            orig_embeddings = self.calculator.model.encode(witnesses, convert_to_numpy=True)
            orig_embeddings = orig_embeddings / np.linalg.norm(orig_embeddings, axis=1, keepdims=True)
            # Add bias as additional constraint
            all_embeddings = np.vstack([orig_embeddings, bias['direction'].reshape(1, -1)])
        else:
            all_embeddings = bias['direction'].reshape(1, -1)

        # Compute new ambiguity with bias
        # We manually compute since we have custom embeddings
        mean_direction = np.mean(all_embeddings, axis=0)
        mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-10)

        if len(all_embeddings) > 1:
            pairwise_angles = np.arccos(np.clip(all_embeddings @ all_embeddings.T, -1, 1))
            np.fill_diagonal(pairwise_angles, 0)
            avg_angle = np.mean(pairwise_angles[np.triu_indices(len(all_embeddings), k=1)])
            orthogonality = avg_angle / (np.pi / 2)
            effective_constraints = len(all_embeddings) * min(orthogonality, 1.0)
            solid_angle = 4 * np.pi / (2 ** effective_constraints)
        else:
            solid_angle = 2 * np.pi

        new_ambiguity = solid_angle / (4 * np.pi)

        return {
            'success': True,
            'baseline_ambiguity': baseline['ambiguity'],
            'new_ambiguity': float(new_ambiguity),
            'reduction': baseline['ambiguity'] - new_ambiguity,
            'reduction_percent': (baseline['ambiguity'] - new_ambiguity) / baseline['ambiguity'] * 100 if baseline['ambiguity'] > 0 else 0,
            'bias_applied': bias_name,
            'bias_reason': bias['reason']
        }

    def generate_bias_disclosure(self) -> str:
        """Generate disclosure statement for all applied biases"""
        if len(self.biases) == 0:
            return "No biases have been applied to this response."

        disclosure = "The following biases have been applied to reduce ambiguity:\n"
        for bias in self.biases:
            disclosure += f"- {bias['name']}: {bias['reason']}\n"
        return disclosure


def test_conservation_law():
    """
    Test that ambiguity ONLY decreases via new witnesses or bias.
    This is the main validation of the Conservation of Ambiguity principle.
    """
    print("\n" + "="*60)
    print("TEST: CONSERVATION OF AMBIGUITY LAW")
    print("="*60)

    # Load model
    print("\nLoading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    calc = AmbiguityCalculator(model)

    results = {
        'monotonicity_tests': [],
        'bias_tests': [],
        'conservation_validated': False
    }

    # Test Case 1: Adding witnesses reduces ambiguity (monotonicity)
    print("\n" + "-"*40)
    print("TEST 1: Monotonicity - Adding Witnesses Reduces Ambiguity")
    print("-"*40)

    test_sequences = [
        # Sequence 1: Object specification
        {
            'name': 'Object Specification',
            'witnesses': [
                ["bike"],
                ["bike", "red"],
                ["bike", "red", "racing"],
                ["bike", "red", "racing", "carbon fiber"]
            ]
        },
        # Sequence 2: Location
        {
            'name': 'Location Query',
            'witnesses': [
                ["restaurant"],
                ["restaurant", "Italian"],
                ["restaurant", "Italian", "downtown"],
                ["restaurant", "Italian", "downtown", "romantic"]
            ]
        },
        # Sequence 3: Abstract concept
        {
            'name': 'Abstract Concept',
            'witnesses': [
                ["love"],
                ["love", "romantic"],
                ["love", "romantic", "long-distance"],
                ["love", "romantic", "long-distance", "reunited"]
            ]
        },
        # Sequence 4: Technical query
        {
            'name': 'Technical Query',
            'witnesses': [
                ["algorithm"],
                ["algorithm", "sorting"],
                ["algorithm", "sorting", "efficient"],
                ["algorithm", "sorting", "efficient", "parallel"]
            ]
        }
    ]

    monotonicity_results = []

    for seq in test_sequences:
        print(f"\n{seq['name']}:")
        ambiguities = []

        for i, W in enumerate(seq['witnesses']):
            amb_result = calc.compute_ambiguity(W)
            ambiguities.append(amb_result['ambiguity'])
            print(f"  W{i+1} = {W}")
            print(f"       Ambiguity: {amb_result['ambiguity']:.4f} (solid angle: {amb_result['solid_angle']:.4f})")

        # Check monotonicity
        is_monotonic = all(ambiguities[i] >= ambiguities[i+1] for i in range(len(ambiguities)-1))
        monotonicity_results.append({
            'name': seq['name'],
            'witnesses': seq['witnesses'],
            'ambiguities': ambiguities,
            'monotonic': is_monotonic,
            'total_reduction': ambiguities[0] - ambiguities[-1]
        })

        status = "✓ MONOTONIC" if is_monotonic else "✗ VIOLATED"
        print(f"  Result: {status} (reduction: {ambiguities[0]:.4f} → {ambiguities[-1]:.4f})")

    results['monotonicity_tests'] = monotonicity_results

    # Test Case 2: Bias injection reduces ambiguity
    print("\n" + "-"*40)
    print("TEST 2: Bias Injection Reduces Ambiguity")
    print("-"*40)

    tracker = BiasTracker(calc)

    # Register some biases
    # Get embedding direction for bias concepts
    bias_concepts = [
        ("positive_sentiment", "happy joyful positive optimistic good", "Preference for positive framing"),
        ("formal_style", "professional formal academic scholarly", "Preference for formal language"),
        ("simple_explanation", "simple easy basic beginner friendly", "Preference for accessibility")
    ]

    for name, concept, reason in bias_concepts:
        bias_embedding = model.encode([concept], convert_to_numpy=True)[0]
        bias_embedding = bias_embedding / np.linalg.norm(bias_embedding)
        tracker.register_bias(name, bias_embedding, reason)

    bias_test_cases = [
        {
            'name': 'Ambiguous sentiment query',
            'witnesses': ["weather", "tomorrow"],
            'bias': 'positive_sentiment'
        },
        {
            'name': 'Style-ambiguous query',
            'witnesses': ["explain", "quantum mechanics"],
            'bias': 'formal_style'
        },
        {
            'name': 'Complexity-ambiguous query',
            'witnesses': ["how does", "neural network", "work"],
            'bias': 'simple_explanation'
        }
    ]

    bias_results = []

    for test in bias_test_cases:
        print(f"\n{test['name']}:")
        print(f"  Witnesses: {test['witnesses']}")

        result = tracker.apply_bias(test['witnesses'], test['bias'])

        if result['success']:
            print(f"  Baseline ambiguity: {result['baseline_ambiguity']:.4f}")
            print(f"  After bias '{test['bias']}': {result['new_ambiguity']:.4f}")
            print(f"  Reduction: {result['reduction_percent']:.1f}%")

            bias_results.append({
                'name': test['name'],
                'witnesses': test['witnesses'],
                'bias': test['bias'],
                'baseline': result['baseline_ambiguity'],
                'after_bias': result['new_ambiguity'],
                'reduction': result['reduction'],
                'reduced': result['reduction'] > 0
            })

    results['bias_tests'] = bias_results

    # Test Case 3: Conservation - no spontaneous decrease without witnesses/bias
    print("\n" + "-"*40)
    print("TEST 3: Conservation - No Spontaneous Decrease")
    print("-"*40)

    # Same witnesses, computed multiple times should give same ambiguity
    conservation_witnesses = ["technology", "future"]
    print(f"\nWitnesses: {conservation_witnesses}")

    measurements = []
    for i in range(5):
        amb = calc.compute_ambiguity(conservation_witnesses)
        measurements.append(amb['ambiguity'])
        print(f"  Measurement {i+1}: {amb['ambiguity']:.6f}")

    variance = np.var(measurements)
    is_conserved = variance < 1e-10  # Should be exactly the same
    print(f"\nVariance across measurements: {variance:.2e}")
    print(f"Conservation: {'✓ CONSERVED' if is_conserved else '✗ NOT CONSERVED'}")

    results['conservation_variance'] = float(variance)
    results['conservation_test_passed'] = is_conserved

    # Overall validation
    all_monotonic = all(r['monotonic'] for r in monotonicity_results)
    all_bias_reduced = all(r['reduced'] for r in bias_results)

    results['conservation_validated'] = all_monotonic and all_bias_reduced and is_conserved

    print("\n" + "="*60)
    print("CONSERVATION LAW VALIDATION SUMMARY")
    print("="*60)
    print(f"Monotonicity tests passed: {sum(1 for r in monotonicity_results if r['monotonic'])}/{len(monotonicity_results)}")
    print(f"Bias reduction tests passed: {sum(1 for r in bias_results if r['reduced'])}/{len(bias_results)}")
    print(f"Conservation test passed: {'Yes' if is_conserved else 'No'}")
    print(f"\nOVERALL: {'✓ VALIDATED' if results['conservation_validated'] else '✗ NOT VALIDATED'}")

    return results


def test_refusal_policy():
    """Test the refusal policy based on ambiguity threshold"""
    print("\n" + "="*60)
    print("TEST: REFUSAL POLICY")
    print("="*60)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    calc = AmbiguityCalculator(model)
    policy = RefusalPolicy(calc, threshold=0.3)

    test_cases = [
        # High ambiguity - should refuse
        [],
        ["thing"],
        ["help"],
        # Medium ambiguity
        ["weather", "today"],
        ["recipe", "dinner"],
        # Low ambiguity - should not refuse
        ["python", "function", "sort", "list", "ascending"],
        ["recipe", "chocolate", "cake", "gluten-free", "vegan"],
    ]

    results = []

    print(f"\nRefusal threshold: {policy.threshold}")
    print("-"*40)

    for witnesses in test_cases:
        result = policy.should_refuse(witnesses)
        results.append({
            'witnesses': witnesses,
            'ambiguity': result['ambiguity'],
            'should_refuse': result['should_refuse']
        })

        status = "REFUSE" if result['should_refuse'] else "RESPOND"
        print(f"\nWitnesses: {witnesses if witnesses else '(none)'}")
        print(f"  Ambiguity: {result['ambiguity']:.4f}")
        print(f"  Decision: {status}")
        if result['should_refuse']:
            explanation = policy.generate_refusal_explanation(witnesses)
            print(f"  Explanation: {explanation}")

    return results


def test_hemisphere_constraint():
    """Test the hemisphere constraint for meaning consistency"""
    print("\n" + "="*60)
    print("TEST: HEMISPHERE CONSTRAINT")
    print("="*60)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    calc = AmbiguityCalculator(model)

    # Test cases: meaning vs witnesses
    test_cases = [
        {
            'meaning': "A fast red racing bicycle",
            'witnesses': ["bike", "red", "racing", "fast"],
            'expected': True  # Should satisfy
        },
        {
            'meaning': "A blue slow cargo truck",
            'witnesses': ["bike", "red", "racing", "fast"],
            'expected': False  # Should violate
        },
        {
            'meaning': "Italian pasta with tomato sauce",
            'witnesses': ["restaurant", "Italian", "pasta"],
            'expected': True
        },
        {
            'meaning': "Japanese sushi restaurant",
            'witnesses': ["restaurant", "Italian", "pasta"],
            'expected': False  # Contradicts Italian
        }
    ]

    results = []

    for test in test_cases:
        meaning_emb = model.encode([test['meaning']], convert_to_numpy=True)[0]
        result = calc.satisfies_hemisphere_constraint(meaning_emb, test['witnesses'])

        results.append({
            'meaning': test['meaning'],
            'witnesses': test['witnesses'],
            'satisfies': result['satisfies'],
            'expected': test['expected'],
            'correct': result['satisfies'] == test['expected'],
            'violations': result['violations']
        })

        print(f"\nMeaning: '{test['meaning']}'")
        print(f"Witnesses: {test['witnesses']}")
        print(f"Satisfies: {result['satisfies']} (expected: {test['expected']})")
        if result['violations']:
            print(f"Violations: {result['violations']}")
        print(f"Min dot product: {result['min_dot_product']:.4f}")

    return results


def visualize_results(conservation_results, refusal_results, hemisphere_results):
    """Create visualization of experiment results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Monotonicity - Ambiguity vs Number of Witnesses
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(conservation_results['monotonicity_tests'])))

    for i, test in enumerate(conservation_results['monotonicity_tests']):
        x = range(1, len(test['ambiguities']) + 1)
        ax1.plot(x, test['ambiguities'], 'o-', color=colors[i],
                label=test['name'], linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Witnesses')
    ax1.set_ylabel('Ambiguity')
    ax1.set_title('Monotonicity: Ambiguity Decreases with More Witnesses')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.6)

    # Plot 2: Bias Effect
    ax2 = axes[0, 1]
    bias_tests = conservation_results['bias_tests']
    x = range(len(bias_tests))
    width = 0.35

    baselines = [t['baseline'] for t in bias_tests]
    after_bias = [t['after_bias'] for t in bias_tests]

    bars1 = ax2.bar([i - width/2 for i in x], baselines, width, label='Before Bias', color='lightcoral')
    bars2 = ax2.bar([i + width/2 for i in x], after_bias, width, label='After Bias', color='lightgreen')

    ax2.set_xlabel('Test Case')
    ax2.set_ylabel('Ambiguity')
    ax2.set_title('Bias Injection Reduces Ambiguity')
    ax2.set_xticks(x)
    ax2.set_xticklabels([t['bias'] for t in bias_tests], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Refusal Policy
    ax3 = axes[1, 0]
    ambiguities = [r['ambiguity'] for r in refusal_results]
    should_refuse = [r['should_refuse'] for r in refusal_results]
    colors_refusal = ['red' if r else 'green' for r in should_refuse]

    bars = ax3.bar(range(len(ambiguities)), ambiguities, color=colors_refusal)
    ax3.axhline(y=0.3, color='black', linestyle='--', label='Threshold (0.3)')

    ax3.set_xlabel('Test Case Index')
    ax3.set_ylabel('Ambiguity')
    ax3.set_title('Refusal Policy: Red = Refuse, Green = Respond')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add witness count labels
    for i, r in enumerate(refusal_results):
        witness_count = len(r['witnesses'])
        ax3.text(i, ambiguities[i] + 0.02, f'W={witness_count}', ha='center', fontsize=8)

    # Plot 4: Hemisphere Constraint Results
    ax4 = axes[1, 1]

    # Create summary of hemisphere tests
    correct = sum(1 for r in hemisphere_results if r['correct'])
    incorrect = len(hemisphere_results) - correct

    labels = ['Correct\nPrediction', 'Incorrect\nPrediction']
    sizes = [correct, incorrect]
    colors_pie = ['lightgreen', 'lightcoral']

    # Filter out zero values
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
    if non_zero:
        labels, sizes, colors_pie = zip(*non_zero)
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie,
                                           autopct='%1.0f%%', startangle=90)
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center')

    ax4.set_title(f'Hemisphere Constraint Validation\n({correct}/{len(hemisphere_results)} correct)')

    plt.tight_layout()
    plt.savefig('experiment8_conservation_ambiguity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[SAVED] Visualization saved to experiment8_conservation_ambiguity.png")


def main():
    """Main entry point for Experiment 8"""
    print("="*70)
    print("EXPERIMENT 8: CONSERVATION OF AMBIGUITY VALIDATION")
    print("="*70)
    print("\nThis experiment validates the Conservation of Ambiguity principle:")
    print("- Ambiguity can ONLY decrease via new witnesses or explicit bias")
    print("- Without these, ambiguity is conserved (no spontaneous decrease)")
    print("="*70)

    # Run all tests
    conservation_results = test_conservation_law()
    refusal_results = test_refusal_policy()
    hemisphere_results = test_hemisphere_constraint()

    # Create visualization
    visualize_results(conservation_results, refusal_results, hemisphere_results)

    # Compile statistics
    stats = {
        'validated': conservation_results['conservation_validated'],
        'monotonicity_pass_rate': sum(1 for r in conservation_results['monotonicity_tests'] if r['monotonic']) / len(conservation_results['monotonicity_tests']),
        'bias_reduction_rate': sum(1 for r in conservation_results['bias_tests'] if r['reduced']) / len(conservation_results['bias_tests']),
        'conservation_variance': conservation_results['conservation_variance'],
        'hemisphere_accuracy': sum(1 for r in hemisphere_results if r['correct']) / len(hemisphere_results),
        'refusal_threshold': 0.3
    }

    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT 8 SUMMARY")
    print("="*70)
    print(f"Conservation of Ambiguity: {'✓ VALIDATED' if stats['validated'] else '✗ NOT VALIDATED'}")
    print(f"  - Monotonicity pass rate: {stats['monotonicity_pass_rate']:.1%}")
    print(f"  - Bias reduction rate: {stats['bias_reduction_rate']:.1%}")
    print(f"  - Conservation variance: {stats['conservation_variance']:.2e}")
    print(f"  - Hemisphere constraint accuracy: {stats['hemisphere_accuracy']:.1%}")
    print("="*70)

    return conservation_results, refusal_results, hemisphere_results, stats


if __name__ == "__main__":
    main()
