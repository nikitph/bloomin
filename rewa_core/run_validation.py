#!/usr/bin/env python3
"""
Rewa Core v1 - Full Validation Runner

Runs all test suites and generates compliance report.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_suite import run_all_tests
from src.core import RewaCore
import json
from datetime import datetime


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║                         REWA CORE v1                                  ║
    ║                                                                       ║
    ║              Policy-Driven Semantic Validation Engine                 ║
    ║                                                                       ║
    ║    "Rewa is not an AI that 'knows.' It is a system that refuses      ║
    ║     to lie under policy constraints."                                 ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Run all test suites
    results = run_all_tests()

    # Generate and print compliance report
    print("\n")
    rewa = RewaCore(log_dir="logs")

    # Do a few sample decisions to populate audit log
    policy_id = rewa.register_policy(
        name="Demo Policy",
        description="Demonstration policy",
        rules=["Verify evidence"],
        prototypes=["verified", "confirmed"],
        threshold=0.4
    )

    sample_docs = [
        ["The product meets quality standards", "Testing confirmed compliance"],
        ["Mixed reviews from users", "Some issues reported"],
        ["Excellent performance", "Well documented"],
    ]

    for docs in sample_docs:
        rewa.process(docs, policy_id)

    report = rewa.get_audit_report()
    print(report)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "test_results": {
            suite_name: {
                "passed": r.passed,
                "failed": r.failed,
                "tests": r.results
            }
            for suite_name, r in results.items()
        },
        "statistics": rewa.get_session_statistics()
    }

    with open("validation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n[SAVED] Results saved to validation_results.json")

    return results


if __name__ == "__main__":
    main()
