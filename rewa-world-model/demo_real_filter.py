#!/usr/bin/env python3
"""
Demo: Real Reasoning Filter

Demonstrates the data-driven filter using topos/geometric machinery.
"""

import sys
sys.path.insert(0, 'src')

from middleware import RealReasoningFilter, FilterConfig

def print_header(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def demo_modifier_detection():
    """Show how the new filter detects semantic modifiers."""
    print_header("DEMO 1: Modifier Detection (Fake vs Real)")

    filter = RealReasoningFilter()

    query = "real diamond ring expensive jewelry"
    chunks = [
        "This authentic diamond ring is certified by GIA and costs $10,000",
        "Fake cubic zirconia rings look similar but cost only $50",
        "Diamond engagement rings are a symbol of commitment"
    ]

    result = filter.verify_context(query, chunks)

    print(f"\nQuery: '{query}'")
    print(f"\nOverall Confidence: {result.overall_confidence:.2%}")

    for i, cv in enumerate(result.verified_chunks):
        status = "PASS" if cv.is_consistent else "FLAG"
        print(f"\nChunk {i} [{status}]:")
        print(f"  Text: {chunks[i][:60]}...")
        print(f"  Consistency: {cv.consistency_score:.2f}")
        print(f"  Modifier Score: {cv.modifier_score:.2f}")
        if cv.flags:
            print(f"  Flags: {cv.flags}")

    if result.global_warnings:
        print("\nGlobal Warnings:")
        for w in result.global_warnings:
            print(f"  - {w}")


def demo_contradiction_detection():
    """Show how the filter detects contradictory information."""
    print_header("DEMO 2: Contradiction Detection")

    filter = RealReasoningFilter()

    query = "weather forecast today"
    chunks = [
        "Today will be sunny with clear skies and temperatures around 75F",
        "Heavy rain and thunderstorms expected throughout the day today",
        "Perfect weather for outdoor activities this afternoon"
    ]

    result = filter.verify_context(query, chunks)

    print(f"\nQuery: '{query}'")
    print(f"\nOverall Confidence: {result.overall_confidence:.2%}")

    for i, cv in enumerate(result.verified_chunks):
        print(f"\nChunk {i}:")
        print(f"  Text: {chunks[i][:60]}...")
        print(f"  Score: {cv.consistency_score * cv.modifier_score:.2f}")

    if result.global_warnings:
        print("\nGlobal Warnings (Contradictions Detected):")
        for w in result.global_warnings:
            print(f"  - {w}")


def demo_feasibility_check():
    """Show how the filter checks query feasibility against indexed docs."""
    print_header("DEMO 3: Query Feasibility Check")

    filter = RealReasoningFilter(FilterConfig(feasibility_threshold=8.0))

    # Index some documents about programming
    documents = [
        {"id": "d1", "text": "Python is a popular programming language for data science"},
        {"id": "d2", "text": "JavaScript enables interactive web applications"},
        {"id": "d3", "text": "Machine learning algorithms process large datasets"}
    ]
    filter.index_documents(documents)

    # Related query (should pass feasibility)
    query1 = "python programming data"
    chunks1 = ["Python excels at data processing tasks"]

    result1 = filter.verify_context(query1, chunks1)
    print(f"\nQuery 1: '{query1}'")
    print(f"  Status: {'FEASIBLE' if result1.overall_confidence > 0 else 'INFEASIBLE'}")
    print(f"  Confidence: {result1.overall_confidence:.2%}")

    # Unrelated query (might fail feasibility)
    query2 = "quantum physics particle accelerator hadron"
    chunks2 = ["Particle physics experiments at CERN"]

    result2 = filter.verify_context(query2, chunks2)
    print(f"\nQuery 2: '{query2}'")
    infeasible = any("INFEASIBLE" in w for w in result2.global_warnings)
    print(f"  Status: {'INFEASIBLE' if infeasible else 'FEASIBLE'}")
    print(f"  Confidence: {result2.overall_confidence:.2%}")
    if result2.global_warnings:
        print(f"  Warnings: {result2.global_warnings[0][:70]}...")


def demo_semantic_analysis():
    """Show detailed semantic analysis capabilities."""
    print_header("DEMO 4: Semantic Analysis Deep Dive")

    filter = RealReasoningFilter()

    query = "electric vehicle battery technology"
    chunks = [
        "Electric vehicles use lithium-ion batteries for propulsion",
        "Gasoline engines have been the dominant technology for decades",  # Different topic
        "Battery technology continues to improve EV range significantly"
    ]

    result = filter.verify_context(query, chunks)

    print(f"\nQuery: '{query}'")
    print(f"\nAnalysis shows which chunks are semantically aligned:")

    for i, cv in enumerate(result.verified_chunks):
        combined_score = cv.consistency_score * cv.modifier_score
        alignment = "ALIGNED" if combined_score > 0.4 else "DIVERGENT"
        print(f"\nChunk {i} [{alignment}]:")
        print(f"  Text: {chunks[i][:55]}...")
        print(f"  Combined Score: {combined_score:.2f}")
        if cv.flags:
            print(f"  Issues: {cv.flags}")

    print("\n[Key Insight]")
    print("  The 'gasoline engines' chunk diverges semantically from the query")
    print("  about electric vehicles, detected via distribution distance.")


def main():
    print("\n" + "#"*70)
    print("#  REWA World Model: Real Reasoning Filter Demo")
    print("#"*70)

    demo_modifier_detection()
    demo_contradiction_detection()
    demo_feasibility_check()
    demo_semantic_analysis()

    print("\n" + "="*70)
    print(" Demo Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
