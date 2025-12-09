#!/usr/bin/env python
"""
Test Semantic REWA API

Tests the new semantic-based REWA implementation with real embeddings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rewa.semantic_api import SemanticREWA, SemanticRewaConfig
from rewa.models import RewaStatus


def test_impossible_queries():
    """Test that impossible queries are correctly detected."""
    print("\n" + "=" * 60)
    print("Test 1: Impossible Query Detection (Semantic)")
    print("=" * 60)

    config = SemanticRewaConfig(impossibility_threshold=0.75)
    rewa = SemanticREWA(config)

    impossible_cases = [
        ("Find a drug that cures cancer with zero side effects",
         [{"id": "1", "text": "MiracleCure eliminates cancer with no side effects."}]),
        ("Perpetual motion machine plans",
         [{"id": "1", "text": "Free energy machine that runs forever."}]),
    ]

    for query, chunks in impossible_cases:
        response = rewa.verify(query, chunks)
        status_ok = response.status == RewaStatus.IMPOSSIBLE
        symbol = "✓" if status_ok else "✗"
        print(f"\n{symbol} Query: '{query[:50]}...'")
        print(f"   Status: {response.status.value}")
        print(f"   Explanation: {response.explanation}")


def test_valid_queries():
    """Test that valid queries are correctly processed."""
    print("\n" + "=" * 60)
    print("Test 2: Valid Query Processing (Semantic)")
    print("=" * 60)

    config = SemanticRewaConfig(impossibility_threshold=0.80)  # Higher threshold
    rewa = SemanticREWA(config)

    valid_cases = [
        ("Find cancer treatment options",
         [{"id": "1", "text": "Chemotherapy is an effective cancer treatment. Side effects include nausea and fatigue."}]),
        ("Safe household items",
         [{"id": "1", "text": "This cleaning product is safe and non-toxic for household use."}]),
    ]

    for query, chunks in valid_cases:
        response = rewa.verify(query, chunks)
        # Should NOT be impossible
        not_impossible = response.status != RewaStatus.IMPOSSIBLE
        symbol = "✓" if not_impossible else "✗"
        print(f"\n{symbol} Query: '{query}'")
        print(f"   Status: {response.status.value}")
        print(f"   Facts: {len(response.safe_facts)}")
        print(f"   Confidence: {response.confidence:.2f}")


def test_negation_sensitivity():
    """Test real vs toy weapon distinction."""
    print("\n" + "=" * 60)
    print("Test 3: Negation Sensitivity - Real vs Toy")
    print("=" * 60)

    rewa = SemanticREWA()

    # Query for real weapon
    query = "Find a real gun for self defense"

    # Toy gun chunk
    toy_response = rewa.verify(query, [
        {"id": "1", "text": "The Nerf N-Strike is a toy gun that shoots foam darts. It is NOT a real weapon."}
    ])

    # Real gun chunk
    real_response = rewa.verify(query, [
        {"id": "1", "text": "The Glock 19 is a semi-automatic pistol used for self-defense. It is a real firearm."}
    ])

    print(f"\nQuery: '{query}'")
    print(f"\nWith TOY chunk:")
    print(f"   Status: {toy_response.status.value}")
    print(f"   Confidence: {toy_response.confidence:.2f}")

    print(f"\nWith REAL chunk:")
    print(f"   Status: {real_response.status.value}")
    print(f"   Confidence: {real_response.confidence:.2f}")

    # Real should have higher confidence or be VALID
    if real_response.confidence > toy_response.confidence or \
       (real_response.status == RewaStatus.VALID and toy_response.status != RewaStatus.VALID):
        print("\n✓ Correctly distinguishes real from toy")
    else:
        print("\n✗ Failed to distinguish real from toy")


def test_contradiction_detection():
    """Test that contradictions are detected."""
    print("\n" + "=" * 60)
    print("Test 4: Contradiction Detection")
    print("=" * 60)

    rewa = SemanticREWA()

    response = rewa.verify("Is this product safe?", [
        {"id": "1", "text": "This product is completely safe and non-toxic."},
        {"id": "2", "text": "Warning: This product is dangerous and can cause harm."},
    ])

    print(f"\nStatus: {response.status.value}")
    print(f"Contradictions found: {len(response.contradictions)}")
    print(f"Explanation: {response.explanation}")


def test_property_extraction():
    """Test semantic property extraction."""
    print("\n" + "=" * 60)
    print("Test 5: Semantic Property Extraction")
    print("=" * 60)

    rewa = SemanticREWA()

    # Test chunks with clear properties
    test_cases = [
        ("Find a dangerous weapon",
         [{"id": "1", "text": "This assault rifle is extremely dangerous and lethal."}]),
        ("Find a safe product",
         [{"id": "1", "text": "This children's toy is safe and harmless."}]),
        ("Find a treatment for cancer",
         [{"id": "1", "text": "This medication effectively treats cancer patients."}]),
    ]

    for query, chunks in test_cases:
        response = rewa.verify(query, chunks)
        print(f"\nQuery: '{query}'")
        print(f"   Status: {response.status.value}")
        print(f"   Facts found: {len(response.safe_facts)}")
        for fact in response.safe_facts[:3]:  # Show first 3 facts
            print(f"      - {fact.predicate}: {fact.value} (conf: {fact.confidence:.2f})")


def run_accuracy_test():
    """Run accuracy test on a set of known queries."""
    print("\n" + "=" * 60)
    print("ACCURACY TEST")
    print("=" * 60)

    rewa = SemanticREWA(SemanticRewaConfig(impossibility_threshold=0.78))

    # Test cases: (query, chunks, expected_status)
    test_cases = [
        # Impossible queries
        ("Cancer cure with no side effects",
         [{"id": "1", "text": "MiracleCure cures cancer with zero side effects."}],
         RewaStatus.IMPOSSIBLE),

        ("Perpetual motion machine",
         [{"id": "1", "text": "Free energy device that runs forever."}],
         RewaStatus.IMPOSSIBLE),

        # Valid queries (should NOT be impossible)
        ("Cancer treatment options",
         [{"id": "1", "text": "Chemotherapy treats cancer. Side effects include nausea."}],
         [RewaStatus.VALID, RewaStatus.INSUFFICIENT_EVIDENCE]),  # Either is acceptable

        ("Find a weapon",
         [{"id": "1", "text": "The Glock 19 is a semi-automatic pistol."}],
         [RewaStatus.VALID, RewaStatus.INSUFFICIENT_EVIDENCE]),

        ("Safe cleaning product",
         [{"id": "1", "text": "This cleaner is safe and non-toxic."}],
         [RewaStatus.VALID, RewaStatus.INSUFFICIENT_EVIDENCE]),
    ]

    passed = 0
    total = len(test_cases)

    for query, chunks, expected in test_cases:
        response = rewa.verify(query, chunks)

        if isinstance(expected, list):
            is_correct = response.status in expected
        else:
            is_correct = response.status == expected

        symbol = "✓" if is_correct else "✗"
        passed += 1 if is_correct else 0

        print(f"\n{symbol} Query: '{query[:40]}...'")
        print(f"   Expected: {expected if isinstance(expected, RewaStatus) else [e.value for e in expected]}")
        print(f"   Got: {response.status.value}")

    print(f"\n{'=' * 60}")
    print(f"Accuracy: {passed}/{total} = {100*passed/total:.1f}%")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("  Semantic REWA API Tests")
    print("=" * 60)

    test_impossible_queries()
    test_valid_queries()
    test_negation_sensitivity()
    test_contradiction_detection()
    test_property_extraction()
    run_accuracy_test()

    print("\n" + "=" * 60)
    print("  Tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
