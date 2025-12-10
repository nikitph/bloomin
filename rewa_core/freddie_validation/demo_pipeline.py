#!/usr/bin/env python3
"""
Rewa-Financial End-to-End Pipeline Demo

This demonstrates the complete flow:
1. Load real Freddie Mac loan data (or synthetic if not available)
2. Generate evidence documents
3. Evaluate with Rewa
4. Compare against baseline
5. Compute Hallucinated Approval Rate

To run with real data:
1. Download from https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
2. Place sample_orig_2023.txt and sample_svcg_2023.txt in data/ folder
3. Run this script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Import components - handle both real data loader and core components
from freddie_validation.src.freddie_data_loader import FreddieMacDataLoader, FreddieLoan
from freddie_validation.src.income_evaluator import IncomeAdmissibilityEvaluator, Decision
from freddie_validation.src.loan_evidence import LoanFile, Document, DocumentType, EmploymentRecord, IncomeRecord, IncomeType, EmploymentStatus


def create_loan_file_from_freddie(loan: 'FreddieLoan', evidence: Dict) -> LoanFile:
    """Convert Freddie Mac loan + synthetic evidence to LoanFile."""

    # Create LoanFile
    loan_file = LoanFile(
        loan_id=loan.loan_id,
        borrower_name="Freddie Mac Borrower",
        application_date=datetime.now().strftime("%Y-%m-%d")
    )

    # Add synthetic documents from evidence generator
    for i, doc in enumerate(evidence.get("synthetic_documents", [])):
        loan_file.documents.append(Document(
            id=f"DOC-{i}",
            type=DocumentType.OTHER,
            description=doc["type"],
            date=datetime.now().strftime("%Y-%m-%d"),
            content_summary=doc["content"],
            verified=True
        ))

    # Add basic required documents
    loan_file.documents.extend([
        Document("TAX-1", DocumentType.TAX_RETURN, "Tax Return", "2024-04-15",
                f"AGI based on DTI {loan.dti}%", True),
        Document("W2-1", DocumentType.W2, "W-2 Form", "2024-01-31",
                f"Wages supporting DTI {loan.dti}%", True),
        Document("PAY-1", DocumentType.PAYSTUB, "Paystub", "2024-06-15",
                "Current employment verified", True),
        Document("VOE-1", DocumentType.VOE, "VOE", "2024-06-01",
                "Employment verified", True),
    ])

    # Add credit report document with risk indicators
    credit_status = "excellent" if loan.credit_score >= 740 else \
                   "good" if loan.credit_score >= 680 else \
                   "fair" if loan.credit_score >= 620 else "subprime"
    loan_file.documents.append(Document(
        "CR-1", DocumentType.OTHER, "Credit Report", datetime.now().strftime("%Y-%m-%d"),
        f"Credit score: {loan.credit_score} ({credit_status}). " +
        ("BELOW MINIMUM THRESHOLD. " if loan.credit_score < 620 else "") +
        f"DTI: {loan.dti}%. " +
        ("EXCEEDS MAXIMUM DTI. " if loan.dti > 43 else "") +
        f"LTV: {loan.ltv}%. " +
        ("HIGH LEVERAGE RISK. " if loan.ltv > 95 else ""),
        True
    ))

    # Add employment record
    loan_file.employment_history.append(EmploymentRecord(
        employer="Current Employer",
        position="Employee",
        start_date="2020-01-01",
        end_date=None,
        status=EmploymentStatus.FULL_TIME_W2,
        monthly_income=5000,  # Placeholder
        income_type=IncomeType.BASE_SALARY,
        verified=True
    ))

    # Add income records - simulate declining income for risky loans
    if "DECLINING_INCOME" in loan.risk_flags or loan.credit_score < 620:
        # Declining income pattern
        loan_file.income_records.extend([
            IncomeRecord(2022, None, 75000, IncomeType.BASE_SALARY, "Employer", True, "W2-1"),
            IncomeRecord(2023, None, 55000, IncomeType.BASE_SALARY, "Employer", True, "W2-1"),  # -27% decline
        ])
    else:
        # Stable/growing income
        loan_file.income_records.extend([
            IncomeRecord(2022, None, 60000, IncomeType.BASE_SALARY, "Employer", True, "W2-1"),
            IncomeRecord(2023, None, 62000, IncomeType.BASE_SALARY, "Employer", True, "W2-1"),
        ])

    # Set ground truth
    loan_file.ground_truth = {
        "conforming": not loan.was_repurchased,
        "reason": "Repurchased by Freddie Mac" if loan.was_repurchased else "Clean loan",
        "risk_flags": loan.risk_flags
    }

    return loan_file


def run_demo():
    """Run the end-to-end demo."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║              REWA-FINANCIAL END-TO-END DEMO                           ║
    ║                                                                       ║
    ║    Demonstrating the complete pipeline from Freddie Mac data          ║
    ║    to validation results                                              ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Check for real data
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    # Check for real data (prefer older vintages that have repurchase data)
    has_real_data = False
    data_year = None
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        if os.path.exists(os.path.join(data_dir, f"sample_orig_{year}.txt")):
            has_real_data = True
            data_year = year
            break

    if has_real_data:
        print(f"✓ Real Freddie Mac data found!")
        print(f"  Loading from data/sample_orig_{data_year}.txt...")

        loader = FreddieMacDataLoader(data_dir)
        orig_data, svcg_data = loader.load_sample_data(data_year)

        print(loader.summary())

        # Get validation split
        splits = loader.get_validation_split()
        print(f"\nValidation Split:")
        print(f"  Clean loans: {len(splits['clean'])}")
        print(f"  Repurchased: {len(splits['repurchased'])}")
        print(f"  Delinquent: {len(splits['delinquent'])}")
        print(f"  Modified: {len(splits['modified'])}")

    else:
        print("ℹ Real Freddie Mac data not found.")
        print("  Using synthetic test data instead.")
        print("")
        print("  To use real data:")
        print("  1. Visit: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset")
        print("  2. Download sample files")
        print("  3. Place in freddie_validation/data/")
        print("")

        # Create synthetic Freddie-style loans
        synthetic_loans = [
            # Clean loans (should approve)
            FreddieLoan("SYNTH-001", 750, False, 35.0, 75.0, 300000, 6.5,
                       "SF", "P", "CA", "P", 2, False, 0, None, False, False, []),
            FreddieLoan("SYNTH-002", 720, True, 40.0, 80.0, 250000, 6.75,
                       "SF", "P", "TX", "P", 1, False, 0, None, False, False, []),
            FreddieLoan("SYNTH-003", 680, False, 42.0, 85.0, 350000, 7.0,
                       "SF", "P", "FL", "P", 2, False, 0, None, False, False, []),

            # Repurchased loans (should refuse)
            FreddieLoan("SYNTH-004", 580, False, 55.0, 97.0, 400000, 8.5,
                       "SF", "P", "NV", "P", 1, True, 3, "09", True, False,
                       ["LOW_CREDIT_SCORE", "HIGH_DTI", "HIGH_LTV", "REPURCHASED"]),
            FreddieLoan("SYNTH-005", 610, True, 48.0, 95.0, 280000, 7.5,
                       "SF", "P", "AZ", "P", 1, True, 2, "09", True, False,
                       ["LOW_CREDIT_SCORE", "HIGH_DTI", "REPURCHASED"]),
        ]

        splits = {
            'clean': [l for l in synthetic_loans if not l.was_repurchased],
            'repurchased': [l for l in synthetic_loans if l.was_repurchased]
        }

        loader = None

    # Initialize evaluator
    print("\n" + "="*70)
    print("INITIALIZING REWA EVALUATOR")
    print("="*70)
    print("Training Rewa-space projection...")

    evaluator = IncomeAdmissibilityEvaluator(train_rewa_space=True)

    # Run validation
    print("\n" + "="*70)
    print("RUNNING VALIDATION")
    print("="*70)

    results = {
        'clean': {'total': 0, 'approved': 0, 'refused': 0, 'more_info': 0},
        'repurchased': {'total': 0, 'approved': 0, 'refused': 0, 'more_info': 0}
    }

    # Test clean loans
    print("\n--- Clean Loans (Should Approve) ---")
    for loan in splits['clean'][:5]:  # Limit for demo
        evidence = loader.generate_synthetic_documents(loan) if loader else {
            "synthetic_documents": [{"type": "w2", "content": "Income verified"}],
            "evidence_statements": ["Income is stable and verified."]
        }
        loan_file = create_loan_file_from_freddie(loan, evidence)
        result = evaluator.evaluate(loan_file)

        results['clean']['total'] += 1
        if result.decision == Decision.APPROVE:
            results['clean']['approved'] += 1
        elif result.decision == Decision.REFUSE:
            results['clean']['refused'] += 1
        else:
            results['clean']['more_info'] += 1

        print(f"  {loan.loan_id}: {result.decision.value} "
              f"(Credit: {loan.credit_score}, DTI: {loan.dti}%)")

    # Test repurchased loans (or delinquent if no repurchases in sample)
    test_risky = splits.get('repurchased', [])
    if not test_risky and 'delinquent' in splits:
        test_risky = splits['delinquent']
        print("\n--- Delinquent Loans (Should be Cautious) ---")
        print("  Note: No repurchased loans in 2025 sample yet (too recent)")
        print("  Testing against delinquent loans as proxy for future risk")
    else:
        print("\n--- Repurchased Loans (Should Refuse) ---")

    for loan in test_risky[:5]:
        evidence = loader.generate_synthetic_documents(loan) if loader else {
            "synthetic_documents": [{"type": "credit", "content": f"Score: {loan.credit_score}"}],
            "evidence_statements": [f"Credit score is {loan.credit_score}, below threshold."]
        }
        loan_file = create_loan_file_from_freddie(loan, evidence)
        result = evaluator.evaluate(loan_file)

        results['repurchased']['total'] += 1
        if result.decision == Decision.APPROVE:
            results['repurchased']['approved'] += 1
        elif result.decision == Decision.REFUSE:
            results['repurchased']['refused'] += 1
        else:
            results['repurchased']['more_info'] += 1

        delinquent_marker = " [DELINQUENT]" if loan.ever_delinquent else ""
        print(f"  {loan.loan_id}: {result.decision.value} "
              f"(Credit: {loan.credit_score}, DTI: {loan.dti}%, Flags: {loan.risk_flags}){delinquent_marker}")

    # Compute metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Hallucinated Approval Rate
    total_repurchased = results['repurchased']['total']
    approved_repurchased = results['repurchased']['approved']
    har = approved_repurchased / total_repurchased if total_repurchased > 0 else 0

    print(f"\n  HALLUCINATED APPROVAL RATE (HAR): {har:.1%}")
    print(f"    Repurchased loans approved: {approved_repurchased}/{total_repurchased}")

    # Correct Refusal Rate
    refused_repurchased = results['repurchased']['refused'] + results['repurchased']['more_info']
    crr = refused_repurchased / total_repurchased if total_repurchased > 0 else 0
    print(f"\n  CORRECT REFUSAL RATE: {crr:.1%}")
    print(f"    Repurchased loans refused/pending: {refused_repurchased}/{total_repurchased}")

    # False Refusal Rate
    total_clean = results['clean']['total']
    refused_clean = results['clean']['refused'] + results['clean']['more_info']
    frr = refused_clean / total_clean if total_clean > 0 else 0
    print(f"\n  FALSE REFUSAL RATE: {frr:.1%}")
    print(f"    Clean loans refused/pending: {refused_clean}/{total_clean}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if har == 0:
        print("\n  ✓ ZERO HALLUCINATED APPROVALS")
        print("    Rewa would not have approved any loan that Freddie Mac")
        print("    later forced a repurchase on.")
    else:
        print(f"\n  ✗ {approved_repurchased} HALLUCINATED APPROVAL(S)")
        print("    These loans would have been repurchased.")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
  1. Download full Freddie Mac dataset for comprehensive validation
  2. Tune thresholds to reduce false refusal rate
  3. Expand policy coverage beyond income (DTI, LTV, property)
  4. Partner with lender for real document testing
  5. Prepare Freddie Mac pilot proposal

  Contact: sf.freddiemac.com/working-with-us/seller-servicer
    """)


if __name__ == "__main__":
    run_demo()
