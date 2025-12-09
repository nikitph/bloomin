#!/usr/bin/env python
"""
REWA Demo Script

Demonstrates the key capabilities of the REWA system:
1. Impossible query detection
2. Negation sensitivity
3. Contradiction handling
4. Semantic validation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rewa import REWA, RewaStatus


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_response(response):
    print(f"  Status: {response.status.value}")
    print(f"  Confidence: {response.confidence:.2f}")
    print(f"  Explanation: {response.explanation}")
    if response.safe_facts:
        print(f"  Safe facts: {len(response.safe_facts)}")
    if response.contradictions:
        print(f"  Contradictions: {len(response.contradictions)}")
    if response.impossibilities:
        print(f"  Impossibilities:")
        for imp in response.impossibilities:
            print(f"    - {imp.reason}")


def demo_impossible_queries():
    """Demonstrate impossible query detection."""
    print_header("Demo 1: Impossible Query Detection")

    rewa = REWA()

    # Test 1: Cancer cure without side effects
    print("\n[Query] 'Find a drug that cures cancer with zero side effects'")
    response = rewa.verify(
        "Find a drug that cures cancer with zero side effects",
        [{"id": "c1", "text": "MiracleCure eliminates all cancer with no side effects."}]
    )
    print_response(response)
    assert response.status == RewaStatus.IMPOSSIBLE

    # Test 2: Perpetual motion
    print("\n[Query] 'Find perpetual motion machine plans'")
    response = rewa.verify(
        "Find perpetual motion machine plans",
        [{"id": "c1", "text": "Build a perpetual motion machine that runs forever."}]
    )
    print_response(response)
    assert response.status == RewaStatus.IMPOSSIBLE

    # Test 3: Valid query (should NOT be impossible)
    print("\n[Query] 'Find cancer treatment options' (valid)")
    response = rewa.verify(
        "Find cancer treatment options",
        [{
            "id": "c1",
            "text": "Chemotherapy is an effective cancer treatment. Side effects "
                   "include nausea and fatigue."
        }]
    )
    print_response(response)
    assert response.status != RewaStatus.IMPOSSIBLE

    print("\n✓ Impossible query detection working!")


def demo_negation_sensitivity():
    """Demonstrate sensitivity to negation."""
    print_header("Demo 2: Negation Sensitivity")

    rewa = REWA()

    # Test 1: Toy gun should NOT satisfy real weapon query
    print("\n[Query] 'Find a real gun for self-defense'")
    print("[Context] Toy gun description")
    response = rewa.verify(
        "Find a real gun for self-defense",
        [{
            "id": "c1",
            "text": "The Nerf N-Strike is a toy gun that shoots foam darts. "
                   "It is NOT a real weapon and is completely safe for children."
        }]
    )
    print_response(response)
    # Should not return as valid with high confidence

    # Test 2: Real gun should satisfy
    print("\n[Query] 'Find a real gun for self-defense'")
    print("[Context] Real gun description")
    response = rewa.verify(
        "Find a real gun for self-defense",
        [{
            "id": "c1",
            "text": "The Glock 19 is a real semi-automatic pistol used for "
                   "self-defense. It is a dangerous firearm."
        }]
    )
    print_response(response)

    print("\n✓ Negation sensitivity working!")


def demo_contradiction_detection():
    """Demonstrate contradiction detection."""
    print_header("Demo 3: Contradiction Detection")

    rewa = REWA()

    # Contradictory information about safety
    print("\n[Query] 'Is Product X safe?'")
    print("[Context] Contradictory claims about safety")
    response = rewa.verify(
        "Is Product X safe?",
        [
            {"id": "c1", "text": "Product X is completely safe and non-toxic."},
            {"id": "c2", "text": "Product X is dangerous and can cause harm."}
        ]
    )
    print_response(response)

    print("\n✓ Contradiction detection working!")


def demo_api_usage():
    """Demonstrate the API usage pattern."""
    print_header("Demo 4: Agent Integration Pattern")

    print("""
    # Typical agent integration:

    from rewa import REWA, RewaStatus

    rewa = REWA()

    # After retrieval from vector DB...
    context = vector_db.query(query)

    response = rewa.verify(
        query=query,
        retrieved_chunks=context
    )

    if response.status == RewaStatus.VALID:
        # Use response.safe_facts for generation
        agent.respond(response.safe_facts)

    elif response.status == RewaStatus.IMPOSSIBLE:
        # Query asks for something impossible
        agent.explain(response.explanation)

    elif response.status == RewaStatus.CONFLICT:
        # Contradictory information found
        agent.ask_clarifying_question()

    elif response.status == RewaStatus.AMBIGUOUS:
        # Query spans multiple semantic domains
        agent.ask_for_clarification(response.ambiguous_charts)

    elif response.status == RewaStatus.INSUFFICIENT_EVIDENCE:
        # Not enough information to validate
        agent.say_unknown()
    """)

    print("✓ API pattern demonstrated!")


def main():
    print("\n" + "=" * 60)
    print("  REWA - Reasoning & Validation Layer")
    print("  Demo Script")
    print("=" * 60)

    demo_impossible_queries()
    demo_negation_sensitivity()
    demo_contradiction_detection()
    demo_api_usage()

    print("\n" + "=" * 60)
    print("  All demos completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
