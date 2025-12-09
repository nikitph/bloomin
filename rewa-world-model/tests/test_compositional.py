#!/usr/bin/env python3
"""
Test compositional reasoning with the section extractor.

These tests verify that the system can now handle the "impossible query"
cases that bag-of-words approaches fail on.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from middleware import RealReasoningFilter, FilterConfig
from topos import SectionExtractor, extract_and_query


def test_red_bike_impossible_query():
    """
    The classic test:
    - Doc A: "A red car and a blue bike"
    - Doc B: "A blue car and a red bike"
    - Query: "red bike"

    Only Doc B should match because "red" and "bike" are bound together.
    """
    print("\n" + "="*60)
    print("TEST: Red Bike Impossible Query")
    print("="*60)

    doc_a = "A red car and a blue bike"
    doc_b = "A blue car and a red bike"
    query = "red bike"

    result_a = extract_and_query(doc_a, query, "doc_a")
    result_b = extract_and_query(doc_b, query, "doc_b")

    print(f"\nQuery: '{query}'")
    print(f"Query predicates: {result_a['query_predicates']}")

    print(f"\nDoc A: '{doc_a}'")
    print(f"  Sections extracted:")
    for s in result_a['sections']:
        preds = [p.predicate for p in s.propositions]
        print(f"    {s.region_id}: {preds}")
    print(f"  Has match: {result_a['has_match']}")

    print(f"\nDoc B: '{doc_b}'")
    print(f"  Sections extracted:")
    for s in result_b['sections']:
        preds = [p.predicate for p in s.propositions]
        print(f"    {s.region_id}: {preds}")
    print(f"  Has match: {result_b['has_match']}")

    # Verify
    if not result_a['has_match'] and result_b['has_match']:
        print("\nPASS: Correctly identified Doc B as match, rejected Doc A")
        return True
    else:
        print("\nFAIL: Incorrect matching")
        return False


def test_cheap_red_bike():
    """
    Hard modifier test:
    - Distractor: "The expensive red car is fast. The cheap blue bike is slow."
    - Target: "The cheap red bike is broken."
    - Query: "cheap red bike"

    Only Target should match.
    """
    print("\n" + "="*60)
    print("TEST: Cheap Red Bike (Hard Modifier)")
    print("="*60)

    distractor = "The expensive red car is fast. The cheap blue bike is slow."
    target = "The cheap red bike is broken."
    query = "cheap red bike"

    result_dist = extract_and_query(distractor, query, "distractor")
    result_tgt = extract_and_query(target, query, "target")

    print(f"\nQuery: '{query}'")
    print(f"Query predicates: {result_dist['query_predicates']}")

    print(f"\nDistractor: '{distractor}'")
    print(f"  Sections extracted:")
    for s in result_dist['sections']:
        preds = [p.predicate for p in s.propositions]
        print(f"    {s.region_id}: {preds}")
    print(f"  Has match: {result_dist['has_match']}")

    print(f"\nTarget: '{target}'")
    print(f"  Sections extracted:")
    for s in result_tgt['sections']:
        preds = [p.predicate for p in s.propositions]
        print(f"    {s.region_id}: {preds}")
    print(f"  Has match: {result_tgt['has_match']}")

    if not result_dist['has_match'] and result_tgt['has_match']:
        print("\nPASS: Correctly rejected distractor, matched target")
        return True
    else:
        print("\nFAIL: Incorrect matching")
        return False


def test_fake_gun_modifier():
    """
    Authenticity modifier test:
    - Doc A: "The real gun is dangerous"
    - Doc B: "The fake gun is safe"
    - Query: "real gun"

    Only Doc A should match.
    """
    print("\n" + "="*60)
    print("TEST: Real vs Fake Gun")
    print("="*60)

    doc_a = "The real gun is dangerous"
    doc_b = "The fake gun is safe"
    query = "real gun"

    result_a = extract_and_query(doc_a, query, "doc_a")
    result_b = extract_and_query(doc_b, query, "doc_b")

    print(f"\nQuery: '{query}'")
    print(f"Query predicates: {result_a['query_predicates']}")

    print(f"\nDoc A: '{doc_a}'")
    print(f"  Sections: {[{p.predicate for p in s.propositions} for s in result_a['sections']]}")
    print(f"  Has match: {result_a['has_match']}")

    print(f"\nDoc B: '{doc_b}'")
    print(f"  Sections: {[{p.predicate for p in s.propositions} for s in result_b['sections']]}")
    print(f"  Has match: {result_b['has_match']}")

    if result_a['has_match'] and not result_b['has_match']:
        print("\nPASS: Correctly matched 'real gun', rejected 'fake gun'")
        return True
    else:
        print("\nFAIL: Incorrect matching")
        return False


def test_filter_compositional_method():
    """
    Test the verify_compositional method on the filter.
    """
    print("\n" + "="*60)
    print("TEST: Filter verify_compositional Method")
    print("="*60)

    filter = RealReasoningFilter()

    query = "red bike"
    chunks = [
        "A red car and a blue bike",  # No match
        "A blue car and a red bike",  # Match
        "The green truck is parked"   # No match
    ]

    result = filter.verify_compositional(query, chunks)

    print(f"\nQuery: '{query}'")
    print(f"Query predicates: {result['query_predicates']}")

    for cr in result['chunk_results']:
        print(f"\nChunk {cr['chunk_index']}: '{cr['text']}'")
        print(f"  Sections: {cr['sections']}")
        print(f"  Matches: {cr['matching_sections']}")
        print(f"  Has compositional match: {cr['has_compositional_match']}")

    print(f"\nMatching chunk indices: {result['matching_chunks']}")
    print(f"Best match: {result['best_match']}")

    # Should only match chunk 1
    if result['matching_chunks'] == [1] and result['best_match'] == 1:
        print("\nPASS: Correctly identified chunk 1 as only match")
        return True
    else:
        print("\nFAIL: Incorrect matching")
        return False


def test_multi_attribute_binding():
    """
    Test with multiple attributes that must all bind.

    Note: "The car is fast" puts "fast" as a predicate adjective (after the noun).
    The extractor currently only handles pre-nominal modifiers.
    So we use "fast expensive red car" where all modifiers are before the noun.
    """
    print("\n" + "="*60)
    print("TEST: Multi-Attribute Binding (fast expensive red car)")
    print("="*60)

    # Pre-nominal modifiers only (all adjectives before noun)
    docs = [
        "The fast expensive blue car is here. The slow cheap red truck is there.",
        "The fast expensive red car arrived today.",
        "Red cars can be expensive. Fast vehicles are popular."
    ]
    query = "fast expensive red car"

    extractor = SectionExtractor()
    query_preds = extractor.query_to_predicates(query)

    print(f"\nQuery: '{query}'")
    print(f"Required predicates: {query_preds}")

    for i, doc in enumerate(docs):
        sections = extractor.extract_sections(doc, f"doc_{i}")
        matches = extractor.find_matching_sections(sections, query_preds)

        print(f"\nDoc {i}: '{doc}'")
        for s in sections:
            preds = [p.predicate for p in s.propositions]
            print(f"  Section: {preds}")
        print(f"  Has full match: {len(matches) > 0}")

    # Doc 1 should be the only match (fast + expensive + red + car in same phrase)
    result_1 = extract_and_query(docs[1], query, "doc_1")

    if result_1['has_match']:
        print("\nPASS: Correctly found multi-attribute binding in Doc 1")
        return True
    else:
        print("\nFAIL: Should have matched Doc 1")
        return False


def test_section_extractor_edge_cases():
    """
    Test edge cases for the section extractor.
    """
    print("\n" + "="*60)
    print("TEST: Section Extractor Edge Cases")
    print("="*60)

    extractor = SectionExtractor()

    # Empty text
    sections = extractor.extract_sections("", "empty")
    print(f"Empty text: {len(sections)} sections")
    assert len(sections) == 0

    # No nouns
    sections = extractor.extract_sections("very quickly and slowly", "no_nouns")
    print(f"No nouns: {len(sections)} sections")
    assert len(sections) == 0

    # Multiple modifiers
    text = "The big fast red car"
    sections = extractor.extract_sections(text, "multi_mod")
    print(f"Multiple modifiers '{text}':")
    for s in sections:
        preds = [p.predicate for p in s.propositions]
        print(f"  {preds}")
    # Should have is_big, is_fast, is_red, is_car all in one section
    assert len(sections) == 1
    preds = {p.predicate for p in sections[0].propositions}
    # All modifiers should be captured
    assert "is_car" in preds
    assert "is_red" in preds
    assert "is_fast" in preds
    assert "is_big" in preds

    # Plural nouns
    text = "red bikes and blue cars"
    sections = extractor.extract_sections(text, "plurals")
    print(f"Plurals '{text}':")
    for s in sections:
        preds = [p.predicate for p in s.propositions]
        print(f"  {preds}")
    # Should normalize bikes->bike, cars->car
    all_preds = set()
    for s in sections:
        for p in s.propositions:
            all_preds.add(p.predicate)
    assert "is_bike" in all_preds
    assert "is_car" in all_preds

    print("\nPASS: All edge cases handled correctly")
    return True


def run_all_tests():
    """Run all compositional tests."""
    print("\n" + "#"*60)
    print("#  Compositional Reasoning Tests")
    print("#"*60)

    results = []

    results.append(("Red Bike Impossible Query", test_red_bike_impossible_query()))
    results.append(("Cheap Red Bike", test_cheap_red_bike()))
    results.append(("Real vs Fake Gun", test_fake_gun_modifier()))
    results.append(("Filter verify_compositional", test_filter_compositional_method()))
    results.append(("Multi-Attribute Binding", test_multi_attribute_binding()))
    results.append(("Edge Cases", test_section_extractor_edge_cases()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = sum(1 for _, p in results if p)
    print(f"\n  Total: {total}/{len(results)} tests passed")

    return all(p for _, p in results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
