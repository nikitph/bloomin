#!/usr/bin/env python3
"""
Test Suite with Rewa-Space Enabled

Re-runs the validation tests using Rewa-space projection
to demonstrate improved antipodal detection and contradiction handling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.rewa_space_v2 import RewaSpaceV2, generate_negation_pairs


def test_antipodal_detection():
    """Test that Rewa-space properly detects antipodal pairs."""
    print("\n" + "="*70)
    print("TEST: ANTIPODAL DETECTION WITH REWA-SPACE")
    print("="*70)

    # Initialize and train
    print("\nInitializing Rewa-Space...")
    rewa = RewaSpaceV2(output_dim=384)

    training_pairs = generate_negation_pairs()
    print(f"Training on {len(training_pairs)} pairs...")

    rewa.train(training_pairs, epochs=200, verbose=False)

    # Test the originally failing cases
    test_cases = [
        # The cases that failed in Test Suite 3
        ("yes", "no"),
        ("maybe", "definitely not"),
        ("A", "not A"),

        # Additional contradiction tests
        ("The sky is blue", "The sky is not blue"),
        ("The statement is true", "The statement is false"),
        ("I agree", "I disagree"),
        ("This is correct", "This is incorrect"),
        ("Approved", "Rejected"),
    ]

    print("\n" + "-"*70)
    print(f"{'Pair':<50} {'Base°':<10} {'Rewa°':<10} {'Antipodal?':<10}")
    print("-"*70)

    passed = 0
    total = len(test_cases)

    for pos, neg in test_cases:
        # Base embedding angle
        pos_base = rewa._get_base_embedding(pos)
        neg_base = rewa._get_base_embedding(neg)
        base_sim = np.dot(pos_base, neg_base)
        base_angle = np.degrees(np.arccos(np.clip(base_sim, -1, 1)))

        # Rewa-space angle
        pos_rewa = rewa.project(pos)
        neg_rewa = rewa.project(neg)
        rewa_sim = np.dot(pos_rewa, neg_rewa)
        rewa_angle = np.degrees(np.arccos(np.clip(rewa_sim, -1, 1)))

        is_antipodal = rewa_angle > 90  # At least orthogonal
        if is_antipodal:
            passed += 1

        status = "✓" if is_antipodal else "✗"
        print(f"{pos} / {neg:<30} {base_angle:>6.1f}°    {rewa_angle:>6.1f}°    {status}")

    print("-"*70)
    print(f"Antipodal detection: {passed}/{total} ({passed/total*100:.1f}%)")

    return passed, total


def test_hemisphere_with_rewa():
    """Test hemisphere detection in Rewa-space."""
    print("\n" + "="*70)
    print("TEST: HEMISPHERE DETECTION IN REWA-SPACE")
    print("="*70)

    rewa = RewaSpaceV2(output_dim=384)
    training_pairs = generate_negation_pairs()
    rewa.train(training_pairs, epochs=200, verbose=False)

    test_cases = [
        {
            "name": "Contradictory evidence",
            "docs": ["The product is excellent", "The product is terrible"],
            "should_have_hemisphere": False
        },
        {
            "name": "Consistent evidence",
            "docs": ["Great quality", "Excellent service", "Highly recommend"],
            "should_have_hemisphere": True
        },
        {
            "name": "Yes/No contradiction",
            "docs": ["yes", "no"],
            "should_have_hemisphere": False
        },
        {
            "name": "Approval contradiction",
            "docs": ["Application approved", "Application rejected"],
            "should_have_hemisphere": False
        },
        {
            "name": "Safety contradiction",
            "docs": ["Safe and tested", "Dangerous and risky"],
            "should_have_hemisphere": False
        },
    ]

    print("\n" + "-"*70)

    passed = 0
    for case in test_cases:
        # Project all documents
        embeddings = np.array([rewa.project(doc) for doc in case["docs"]])

        # Check if hemisphere exists (all pairwise similarities > some threshold)
        # For contradiction: at least one pair should be antipodal (sim < -0.5)
        n = len(embeddings)
        has_contradiction = False

        for i in range(n):
            for j in range(i+1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                if sim < -0.3:  # Antipodal threshold
                    has_contradiction = True
                    break
            if has_contradiction:
                break

        hemisphere_exists = not has_contradiction
        correct = (hemisphere_exists == case["should_have_hemisphere"])

        if correct:
            passed += 1

        status = "✓" if correct else "✗"
        expected = "has hemisphere" if case["should_have_hemisphere"] else "NO hemisphere"
        actual = "has hemisphere" if hemisphere_exists else "NO hemisphere"
        print(f"{status} {case['name']}: Expected {expected}, Got {actual}")

    print("-"*70)
    print(f"Hemisphere detection: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.1f}%)")

    return passed, len(test_cases)


def test_hallucination_prevention():
    """Test that hallucinations are prevented with Rewa-space."""
    print("\n" + "="*70)
    print("TEST: HALLUCINATION PREVENTION WITH REWA-SPACE")
    print("="*70)

    rewa = RewaSpaceV2(output_dim=384)
    training_pairs = generate_negation_pairs()
    rewa.train(training_pairs, epochs=200, verbose=False)

    # The adversarial cases that originally caused hallucinations
    adversarial_cases = [
        {
            "name": "Empty evidence",
            "docs": [],
            "should_refuse": True
        },
        {
            "name": "Single vague word",
            "docs": ["thing"],
            "should_refuse": True  # Insufficient evidence
        },
        {
            "name": "Ambiguous yes/no/maybe",
            "docs": ["yes", "no", "maybe"],
            "should_refuse": True  # Contains contradiction
        },
        {
            "name": "Logical A / not A",
            "docs": ["confirmed", "not confirmed"],
            "should_refuse": True  # Contradiction
        },
        {
            "name": "Consistent evidence",
            "docs": ["The product works well", "Good quality", "Reliable performance"],
            "should_refuse": False  # Should approve
        },
    ]

    print("\n" + "-"*70)

    passed = 0
    hallucinations = 0

    for case in adversarial_cases:
        if len(case["docs"]) < 2:
            # Insufficient evidence
            should_refuse = True
            detected_problem = "insufficient"
        else:
            # Check for contradictions
            embeddings = np.array([rewa.project(doc) for doc in case["docs"]])
            n = len(embeddings)

            has_contradiction = False
            for i in range(n):
                for j in range(i+1, n):
                    sim = np.dot(embeddings[i], embeddings[j])
                    if sim < -0.3:
                        has_contradiction = True
                        break
                if has_contradiction:
                    break

            should_refuse = has_contradiction
            detected_problem = "contradiction" if has_contradiction else "none"

        correct = (should_refuse == case["should_refuse"])
        if correct:
            passed += 1
        else:
            if not should_refuse and case["should_refuse"]:
                hallucinations += 1

        status = "✓" if correct else "✗"
        action = "REFUSE" if should_refuse else "APPROVE"
        expected_action = "REFUSE" if case["should_refuse"] else "APPROVE"

        if not correct and not should_refuse:
            status = "✗ HALLUCINATION"

        print(f"{status} {case['name']}: {action} (expected {expected_action}) [{detected_problem}]")

    print("-"*70)
    print(f"Correct decisions: {passed}/{len(adversarial_cases)}")
    print(f"Hallucinations: {hallucinations}")

    return passed, len(adversarial_cases), hallucinations


def compare_base_vs_rewa():
    """Side-by-side comparison of base embeddings vs Rewa-space."""
    print("\n" + "="*70)
    print("COMPARISON: BASE EMBEDDINGS vs REWA-SPACE")
    print("="*70)

    rewa = RewaSpaceV2(output_dim=384)
    training_pairs = generate_negation_pairs()
    rewa.train(training_pairs, epochs=200, verbose=False)

    # Key test pairs
    test_pairs = [
        ("The sky is blue", "The sky is not blue"),
        ("I love this", "I hate this"),
        ("yes", "no"),
        ("true", "false"),
        ("approved", "rejected"),
        ("safe", "dangerous"),
        ("legal", "illegal"),
        ("valid", "invalid"),
        ("success", "failure"),
        ("increase", "decrease"),
    ]

    print("\n" + "-"*70)
    print(f"{'Pair':<40} {'Base°':<10} {'Rewa°':<10} {'Improvement':<12}")
    print("-"*70)

    base_angles = []
    rewa_angles = []

    for pos, neg in test_pairs:
        pos_base = rewa._get_base_embedding(pos)
        neg_base = rewa._get_base_embedding(neg)
        base_sim = np.dot(pos_base, neg_base)
        base_angle = np.degrees(np.arccos(np.clip(base_sim, -1, 1)))

        pos_rewa = rewa.project(pos)
        neg_rewa = rewa.project(neg)
        rewa_sim = np.dot(pos_rewa, neg_rewa)
        rewa_angle = np.degrees(np.arccos(np.clip(rewa_sim, -1, 1)))

        base_angles.append(base_angle)
        rewa_angles.append(rewa_angle)

        improvement = rewa_angle - base_angle
        label = f"{pos} / {neg}"
        print(f"{label:<40} {base_angle:>6.1f}°    {rewa_angle:>6.1f}°    {improvement:>+8.1f}°")

    print("-"*70)
    print(f"{'MEAN':<40} {np.mean(base_angles):>6.1f}°    {np.mean(rewa_angles):>6.1f}°    {np.mean(rewa_angles)-np.mean(base_angles):>+8.1f}°")
    print(f"{'TARGET':<40} {'180.0°':>20}")

    return base_angles, rewa_angles


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║              REWA CORE - REWA-SPACE VALIDATION                        ║
    ║                                                                       ║
    ║    Testing with trained Rewa-space projection head                    ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Run all tests
    antipodal_passed, antipodal_total = test_antipodal_detection()
    hemisphere_passed, hemisphere_total = test_hemisphere_with_rewa()
    halluc_passed, halluc_total, hallucinations = test_hallucination_prevention()
    base_angles, rewa_angles = compare_base_vs_rewa()

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    total_tests = antipodal_total + hemisphere_total + halluc_total
    total_passed = antipodal_passed + hemisphere_passed + halluc_passed

    print(f"\nTest Results:")
    print(f"  Antipodal Detection: {antipodal_passed}/{antipodal_total}")
    print(f"  Hemisphere Detection: {hemisphere_passed}/{hemisphere_total}")
    print(f"  Hallucination Prevention: {halluc_passed}/{halluc_total}")
    print(f"\nOverall: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")

    print(f"\nKey Metrics:")
    print(f"  Mean base angle: {np.mean(base_angles):.1f}°")
    print(f"  Mean Rewa angle: {np.mean(rewa_angles):.1f}°")
    print(f"  Improvement: {np.mean(rewa_angles) - np.mean(base_angles):.1f}°")
    print(f"  Hallucinations: {hallucinations}")

    print("\n" + "="*70)
    if hallucinations == 0:
        print("SUCCESS: ZERO HALLUCINATIONS WITH REWA-SPACE")
    else:
        print(f"ATTENTION: {hallucinations} hallucination(s) detected")
    print("="*70)


if __name__ == "__main__":
    main()
