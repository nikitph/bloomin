#!/usr/bin/env python
"""
Test REWA with Real Embeddings

This tests the system using actual semantic embeddings instead of
brittle regex patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from rewa.embeddings import Embedder, SemanticMatcher, SemanticNegationDetector


def test_embedder_basics():
    """Test basic embedder functionality."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Embedder")
    print("=" * 60)

    embedder = Embedder()

    # Test embedding
    emb1 = embedder.embed("gun for self defense")
    emb2 = embedder.embed("firearm for protection")
    emb3 = embedder.embed("toy for children")

    print(f"Embedding dimension: {len(emb1)}")
    print(f"Normalized: {np.isclose(np.linalg.norm(emb1), 1.0)}")

    # Similar meanings should have high similarity
    sim_related = embedder.similarity("gun for self defense", "firearm for protection")
    sim_unrelated = embedder.similarity("gun for self defense", "toy for children")

    print(f"\nSimilarity tests:")
    print(f"  'gun for self defense' vs 'firearm for protection': {sim_related:.3f}")
    print(f"  'gun for self defense' vs 'toy for children': {sim_unrelated:.3f}")

    assert sim_related > sim_unrelated, "Related concepts should be more similar"
    print("\n✓ Related concepts have higher similarity")


def test_type_detection():
    """Test semantic type detection."""
    print("\n" + "=" * 60)
    print("Test 2: Semantic Type Detection")
    print("=" * 60)

    matcher = SemanticMatcher()

    test_cases = [
        ("The Glock 19 is a semi-automatic pistol", ["Weapon"]),
        ("This Nerf blaster shoots foam darts", ["Toy"]),
        ("Ibuprofen reduces pain and inflammation", ["Drug"]),
        ("Cancer is a disease of uncontrolled cell growth", ["MedicalCondition"]),
        ("The iPhone charger provides 20W power", ["Device"]),
    ]

    for text, expected_types in test_cases:
        detected = matcher.detect_types(text, threshold=0.4)
        print(f"\n'{text[:50]}...'")
        print(f"  Detected: {detected}")
        print(f"  Expected: {expected_types}")

        # Check if expected types are detected
        for expected in expected_types:
            if expected in detected:
                print(f"  ✓ Found {expected}")
            else:
                print(f"  ✗ Missing {expected}")


def test_property_detection():
    """Test semantic property detection."""
    print("\n" + "=" * 60)
    print("Test 3: Semantic Property Detection")
    print("=" * 60)

    matcher = SemanticMatcher()

    test_cases = [
        ("This weapon is extremely dangerous and lethal", {"dangerous": True}),
        ("This is completely safe and harmless for children", {"dangerous": False}),
        ("This is a toy gun, not a real weapon", {"is_toy": True, "is_real": False}),
        ("This medicine has no side effects", {"has_side_effects": False}),
        ("This treatment causes significant side effects", {"has_side_effects": True}),
        ("This drug cures cancer effectively", {"cures_cancer": True}),
    ]

    for text, expected_props in test_cases:
        detected = matcher.detect_properties(text, threshold=0.4)
        print(f"\n'{text}'")
        print(f"  Detected: {detected}")
        print(f"  Expected: {expected_props}")

        for prop, expected_val in expected_props.items():
            if prop in detected:
                actual_val, conf = detected[prop]
                match = actual_val == expected_val
                symbol = "✓" if match else "✗"
                print(f"  {symbol} {prop}: expected {expected_val}, got {actual_val} (conf: {conf:.2f})")
            else:
                print(f"  ✗ {prop}: not detected")


def test_impossibility_detection():
    """Test semantic impossibility detection."""
    print("\n" + "=" * 60)
    print("Test 4: Semantic Impossibility Detection")
    print("=" * 60)

    matcher = SemanticMatcher()

    # These should be flagged as impossible
    impossible_queries = [
        "Find a drug that cures cancer with zero side effects",
        "Cancer treatment with no adverse effects",
        "Perpetual motion machine plans",
        "Device that generates infinite free energy",
        "A machine that travels faster than light",
        "Show me a square circle",
    ]

    # These should NOT be flagged
    possible_queries = [
        "Find cancer treatment options",
        "Energy efficient devices",
        "Fast transportation methods",
        "Geometric shapes",
    ]

    print("\nShould be IMPOSSIBLE:")
    for query in impossible_queries:
        results = matcher.check_impossibility(query, threshold=0.5)
        if results:
            print(f"  ✓ '{query[:40]}...'")
            print(f"    Reason: {results[0][0]} (conf: {results[0][1]:.2f})")
        else:
            print(f"  ✗ '{query[:40]}...' NOT DETECTED")

    print("\nShould be POSSIBLE (no detection):")
    for query in possible_queries:
        results = matcher.check_impossibility(query, threshold=0.5)
        if not results:
            print(f"  ✓ '{query[:40]}...' correctly not flagged")
        else:
            print(f"  ✗ '{query[:40]}...' incorrectly flagged: {results[0][0]}")


def test_negation_detection():
    """Test semantic negation detection."""
    print("\n" + "=" * 60)
    print("Test 5: Semantic Negation Detection")
    print("=" * 60)

    detector = SemanticNegationDetector()
    detector.precompute()

    test_cases = [
        ("This gun is dangerous", "dangerous", False),
        ("This gun is not dangerous at all", "dangerous", True),
        ("This is a real firearm", "real", False),
        ("This is not a real weapon, just a toy", "real", True),
        ("This medicine has side effects", "side effects", False),
        ("This medicine has no side effects", "side effects", True),
    ]

    for text, concept, expected_negated in test_cases:
        is_negated, confidence = detector.detect_negation(text, concept)
        match = is_negated == expected_negated
        symbol = "✓" if match else "✗"
        print(f"\n'{text}'")
        print(f"  Concept: {concept}")
        print(f"  {symbol} Negated: {is_negated} (expected: {expected_negated}, conf: {confidence:.2f})")


def test_semantic_similarity_for_retrieval():
    """Test that semantic similarity works for retrieval scenarios."""
    print("\n" + "=" * 60)
    print("Test 6: Retrieval Scenario Simulation")
    print("=" * 60)

    embedder = Embedder()

    query = "Find a weapon for self defense"

    chunks = [
        {"id": 1, "text": "The Glock 19 is a reliable pistol commonly used for personal protection."},
        {"id": 2, "text": "Nerf guns are fun toys that shoot foam darts for children."},
        {"id": 3, "text": "Pepper spray is a non-lethal self-defense option."},
        {"id": 4, "text": "The apple tree produces delicious fruit in autumn."},
        {"id": 5, "text": "Combat knives are designed for self-defense and survival."},
    ]

    query_emb = embedder.embed(query)

    # Score all chunks
    scored = []
    for chunk in chunks:
        chunk_emb = embedder.embed(chunk["text"])
        sim = float(np.dot(query_emb, chunk_emb))
        scored.append((chunk, sim))

    # Sort by similarity
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"\nQuery: '{query}'")
    print("\nRanked chunks:")
    for chunk, sim in scored:
        print(f"  {sim:.3f}: [{chunk['id']}] {chunk['text'][:60]}...")

    # The weapon-related chunks should be ranked higher
    top_ids = [chunk["id"] for chunk, _ in scored[:3]]
    weapon_ids = [1, 3, 5]  # Expected top results

    overlap = len(set(top_ids) & set(weapon_ids))
    print(f"\nTop 3 IDs: {top_ids}")
    print(f"Expected weapon IDs: {weapon_ids}")
    print(f"Overlap: {overlap}/3")

    if overlap >= 2:
        print("✓ Semantic retrieval working correctly")
    else:
        print("✗ Semantic retrieval needs tuning")


def test_real_vs_toy_distinction():
    """Critical test: Can we distinguish real weapons from toys?"""
    print("\n" + "=" * 60)
    print("Test 7: Real vs Toy Weapon Distinction (Critical)")
    print("=" * 60)

    embedder = Embedder()

    query = "real gun for self defense"

    real_weapon_chunks = [
        "The Glock 19 is a semi-automatic pistol used by law enforcement.",
        "This .45 caliber handgun is designed for personal protection.",
        "The AR-15 is a real rifle used for sport shooting and defense.",
    ]

    toy_chunks = [
        "The Nerf N-Strike is a toy that shoots foam darts safely.",
        "This plastic water gun is perfect for summer fun.",
        "Airsoft guns are toys that shoot plastic BBs for recreation.",
    ]

    query_emb = embedder.embed(query)

    print(f"\nQuery: '{query}'")

    print("\nReal weapon chunks (should score HIGH):")
    for text in real_weapon_chunks:
        chunk_emb = embedder.embed(text)
        sim = float(np.dot(query_emb, chunk_emb))
        print(f"  {sim:.3f}: {text[:60]}...")

    print("\nToy chunks (should score LOWER):")
    for text in toy_chunks:
        chunk_emb = embedder.embed(text)
        sim = float(np.dot(query_emb, chunk_emb))
        print(f"  {sim:.3f}: {text[:60]}...")

    # Compute average similarities
    real_avg = np.mean([
        float(np.dot(query_emb, embedder.embed(t)))
        for t in real_weapon_chunks
    ])
    toy_avg = np.mean([
        float(np.dot(query_emb, embedder.embed(t)))
        for t in toy_chunks
    ])

    print(f"\nAverage similarity - Real: {real_avg:.3f}, Toy: {toy_avg:.3f}")

    if real_avg > toy_avg:
        print("✓ Real weapons score higher than toys (correct!)")
    else:
        print("✗ Toys score higher than real weapons (problem!)")


def main():
    print("\n" + "=" * 60)
    print("  REWA - Testing with Real Embeddings")
    print("=" * 60)

    print("\nLoading embedding model...")
    embedder = Embedder()
    _ = embedder.embed("warmup")  # Warm up the model
    print("Model loaded!")

    test_embedder_basics()
    test_type_detection()
    test_property_detection()
    test_impossibility_detection()
    test_negation_detection()
    test_semantic_similarity_for_retrieval()
    test_real_vs_toy_distinction()

    print("\n" + "=" * 60)
    print("  All embedding tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
