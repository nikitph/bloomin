#!/usr/bin/env python3
"""
Demo: Compositional Reasoning with Section Extraction

This shows the complete pipeline:
1. Raw text comes in
2. Section extractor identifies noun phrases with their bound modifiers
3. Topos logic checks which sections match the query predicates
4. Only documents with matching compositional structure are returned

This solves the "impossible query" problem that bag-of-words approaches fail on.
"""

import sys
sys.path.insert(0, 'src')

from middleware import RealReasoningFilter
from topos import extract_and_query


def demo_red_bike():
    """The classic test case that breaks bag-of-words search."""
    print("\n" + "="*70)
    print(" DEMO: The 'Impossible Query' - Red Bike")
    print("="*70)

    print("""
The Problem:
  Doc A: "A red car and a blue bike"
  Doc B: "A blue car and a red bike"
  Query: "red bike"

Bag-of-words sees: Both docs have {red, blue, car, bike}
  -> Same score, can't distinguish!

Compositional reasoning sees:
  Doc A: [red+car], [blue+bike]  -> No section has red+bike
  Doc B: [blue+car], [red+bike]  -> Section 2 has red+bike!
""")

    doc_a = "A red car and a blue bike"
    doc_b = "A blue car and a red bike"
    query = "red bike"

    result_a = extract_and_query(doc_a, query, "doc_a")
    result_b = extract_and_query(doc_b, query, "doc_b")

    print(f"Query: '{query}' -> predicates: {result_a['query_predicates']}")

    print(f"\nDoc A: '{doc_a}'")
    for s in result_a['sections']:
        preds = [p.predicate for p in s.propositions]
        print(f"  Section: {preds}")
    print(f"  MATCH: {result_a['has_match']}")

    print(f"\nDoc B: '{doc_b}'")
    for s in result_b['sections']:
        preds = [p.predicate for p in s.propositions]
        print(f"  Section: {preds}")
    print(f"  MATCH: {result_b['has_match']}")

    if result_b['has_match'] and not result_a['has_match']:
        print("\n>> CORRECT: Only Doc B matches 'red bike'")


def demo_modifier_binding():
    """Test that modifiers stay bound to their nouns."""
    print("\n" + "="*70)
    print(" DEMO: Modifier Binding - Fake vs Real")
    print("="*70)

    print("""
The Problem:
  User asks: "real gun"
  Retrieved: "The fake gun is safe for children"

Bag-of-words sees: Has "gun" -> match!
Compositional sees: [fake+gun] != [real+gun] -> reject!
""")

    query = "real gun"
    docs = [
        "The real gun is dangerous and requires a license",
        "The fake gun is a toy made of plastic",
        "Guns can be real or fake depending on material"
    ]

    print(f"Query: '{query}'")

    filter = RealReasoningFilter()
    result = filter.verify_compositional(query, docs)

    print(f"Query predicates: {result['query_predicates']}")

    for cr in result['chunk_results']:
        print(f"\nDoc: '{cr['text']}'")
        for s in cr['sections']:
            print(f"  Section: {s['predicates']}")
        match = "YES" if cr['has_compositional_match'] else "NO"
        print(f"  Compositional match: {match}")

    print(f"\nMatching docs: {result['matching_chunks']}")
    if result['matching_chunks'] == [0]:
        print(">> CORRECT: Only first doc (real gun) matches")


def demo_attribute_swap():
    """The adversarial attribute swap test."""
    print("\n" + "="*70)
    print(" DEMO: Attribute Swap Attack")
    print("="*70)

    print("""
The Attack:
  Query: "cheap red bike"
  Adversarial doc: "The expensive red car is fast. The cheap blue bike is slow."

  This doc has ALL the query words: cheap, red, bike
  But they're bound to DIFFERENT objects!
    - expensive+red -> car
    - cheap+blue -> bike

  Bag-of-words: FOOLED (all words present)
  Compositional: REJECTS (no section has cheap+red+bike)
""")

    query = "cheap red bike"
    adversarial = "The expensive red car is fast. The cheap blue bike is slow."
    target = "The cheap red bike needs repairs."

    result_adv = extract_and_query(adversarial, query, "adversarial")
    result_tgt = extract_and_query(target, query, "target")

    print(f"Query: '{query}' -> {result_adv['query_predicates']}")

    print(f"\nAdversarial: '{adversarial}'")
    for s in result_adv['sections']:
        preds = [p.predicate for p in s.propositions]
        print(f"  Section: {preds}")
    print(f"  MATCH: {result_adv['has_match']}")

    print(f"\nTarget: '{target}'")
    for s in result_tgt['sections']:
        preds = [p.predicate for p in s.propositions]
        print(f"  Section: {preds}")
    print(f"  MATCH: {result_tgt['has_match']}")

    if result_tgt['has_match'] and not result_adv['has_match']:
        print("\n>> CORRECT: Adversarial rejected, target accepted")


def demo_filter_integration():
    """Show the filter's verify_compositional method."""
    print("\n" + "="*70)
    print(" DEMO: Filter Integration")
    print("="*70)

    filter = RealReasoningFilter()

    query = "blue car"
    chunks = [
        "The red car is parked outside.",
        "A blue bike and a green truck.",
        "The fast blue car won the race.",
        "Cars come in many colors including blue."
    ]

    print(f"Query: '{query}'")
    print(f"Chunks: {len(chunks)} documents\n")

    result = filter.verify_compositional(query, chunks)

    for cr in result['chunk_results']:
        match = "MATCH" if cr['has_compositional_match'] else "     "
        print(f"[{match}] '{cr['text'][:50]}...'")
        if cr['matching_sections']:
            print(f"         Matched section: {cr['matching_sections'][0]['predicates']}")

    print(f"\nBest match: Chunk {result['best_match']}")
    print(f"  -> '{chunks[result['best_match']]}'")


def main():
    print("\n" + "#"*70)
    print("#  Compositional Reasoning Demo")
    print("#  Solves the 'impossible query' problem via section extraction")
    print("#"*70)

    demo_red_bike()
    demo_modifier_binding()
    demo_attribute_swap()
    demo_filter_integration()

    print("\n" + "="*70)
    print(" KEY INSIGHT")
    print("="*70)
    print("""
Bag-of-words and embeddings see documents as unstructured token soup.
They can't distinguish:
  - "red car, blue bike" from "blue car, red bike"
  - "real gun" from "fake gun"
  - "cheap red bike" from "expensive red car + cheap blue bike"

Section extraction + topos logic preserves the BINDING between
modifiers and nouns. Queries match only when predicates appear
together in the SAME section - not just anywhere in the document.

This is compositional reasoning: the meaning of "red bike" depends
on "red" and "bike" being bound together, not just co-occurring.
""")


if __name__ == "__main__":
    main()
