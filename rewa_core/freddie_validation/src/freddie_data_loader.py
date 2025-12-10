"""
Freddie Mac Public Dataset Loader

Loads and processes the Freddie Mac Single-Family Loan-Level Dataset
for validation against real loan outcomes.

Dataset: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class ZeroBalanceCode(Enum):
    """Reason loan reached zero balance."""
    PREPAID_OR_MATURED = "01"
    THIRD_PARTY_SALE = "02"
    SHORT_SALE = "03"
    REO_DISPOSITION = "06"
    REPURCHASE = "09"  # KEY - Freddie made lender buy it back
    OTHER = "96"
    UNKNOWN = "00"


@dataclass
class FreddieLoan:
    """A loan from the Freddie Mac dataset."""
    loan_id: str

    # Borrower characteristics
    credit_score: int
    first_time_buyer: bool

    # Loan characteristics
    dti: float  # Debt-to-Income ratio
    ltv: float  # Loan-to-Value ratio
    original_upb: float  # Original unpaid principal balance
    interest_rate: float

    # Property
    property_type: str
    occupancy_status: str
    property_state: str

    # Loan details
    loan_purpose: str  # Purchase, Refinance, etc.
    num_borrowers: int

    # Outcomes (for validation)
    ever_delinquent: bool
    max_delinquency: int  # Max months delinquent
    zero_balance_code: Optional[str]
    was_repurchased: bool  # KEY VALIDATION METRIC
    was_modified: bool

    # Computed risk factors
    risk_flags: List[str]


class FreddieMacDataLoader:
    """
    Loads and processes Freddie Mac loan-level data.

    Column definitions from Freddie Mac documentation:
    https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf
    """

    # Column specifications for origination file (32 columns as of 2025)
    ORIG_COLUMNS = [
        'credit_score', 'first_payment_date', 'first_time_buyer',
        'maturity_date', 'msa', 'mi_pct', 'num_units', 'occupancy_status',
        'cltv', 'dti', 'original_upb', 'ltv', 'interest_rate', 'channel',
        'prepay_penalty', 'amortization_type', 'property_state',
        'property_type', 'postal_code', 'loan_sequence_number', 'loan_purpose',
        'original_loan_term', 'num_borrowers', 'seller_name', 'servicer_name',
        'super_conforming', 'pre_harp_seq', 'program_indicator',
        'harp_indicator', 'num_borrowers_2', 'high_balance_indicator', 'unknown_col'
    ]

    # Column specifications for performance/servicing file (32 columns as of 2025)
    SVCG_COLUMNS = [
        'loan_sequence_number', 'monthly_reporting_period',
        'current_upb', 'current_delinquency', 'loan_age',
        'remaining_months_maturity', 'repurchase_flag', 'modification_flag',
        'zero_balance_code', 'zero_balance_date', 'current_interest_rate',
        'current_deferred_upb', 'due_date_last_payment', 'mi_recoveries',
        'net_sales_proceeds', 'non_mi_recoveries', 'expenses',
        'legal_costs', 'maintenance_costs', 'taxes_insurance',
        'misc_expenses', 'actual_loss', 'modification_cost',
        'step_modification', 'deferred_payment_mod', 'eltv',
        'zero_balance_removal_upb', 'delinquent_interest',
        'col_29', 'col_30', 'col_31', 'col_32'
    ]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.orig_data = None
        self.svcg_data = None

    def load_sample_data(self, year: int = 2023) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load sample origination and servicing data.

        Args:
            year: Year of sample data

        Returns:
            (origination_df, servicing_df)
        """
        orig_file = os.path.join(self.data_dir, f"sample_orig_{year}.txt")
        svcg_file = os.path.join(self.data_dir, f"sample_svcg_{year}.txt")

        if not os.path.exists(orig_file):
            raise FileNotFoundError(
                f"Origination file not found: {orig_file}\n"
                f"Download from: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset"
            )

        # Load origination data
        self.orig_data = pd.read_csv(
            orig_file,
            sep='|',
            names=self.ORIG_COLUMNS,
            dtype=str,
            na_values=['']
        )

        # Load servicing data if available
        if os.path.exists(svcg_file):
            self.svcg_data = pd.read_csv(
                svcg_file,
                sep='|',
                names=self.SVCG_COLUMNS,
                dtype=str,
                na_values=['']
            )

        return self.orig_data, self.svcg_data

    def process_loans(self, limit: int = None) -> List[FreddieLoan]:
        """
        Process loaded data into FreddieLoan objects.

        Args:
            limit: Maximum number of loans to process

        Returns:
            List of FreddieLoan objects
        """
        if self.orig_data is None:
            raise ValueError("No data loaded. Call load_sample_data first.")

        loans = []

        # Merge origination with performance outcomes
        if self.svcg_data is not None:
            # Get final status for each loan
            final_status = self.svcg_data.groupby('loan_sequence_number').last().reset_index()
            merged = self.orig_data.merge(
                final_status[['loan_sequence_number', 'current_delinquency',
                             'repurchase_flag', 'modification_flag', 'zero_balance_code']],
                on='loan_sequence_number',
                how='left'
            )
        else:
            merged = self.orig_data.copy()
            merged['current_delinquency'] = '0'
            merged['repurchase_flag'] = 'N'
            merged['modification_flag'] = 'N'
            merged['zero_balance_code'] = None

        if limit:
            merged = merged.head(limit)

        for _, row in merged.iterrows():
            try:
                loan = self._row_to_loan(row)
                loans.append(loan)
            except Exception as e:
                # Skip malformed rows
                continue

        return loans

    def _row_to_loan(self, row: pd.Series) -> FreddieLoan:
        """Convert a DataFrame row to FreddieLoan."""

        # Parse values with defaults
        def safe_int(val, default=0):
            try:
                return int(float(val)) if pd.notna(val) else default
            except:
                return default

        def safe_float(val, default=0.0):
            try:
                return float(val) if pd.notna(val) else default
            except:
                return default

        credit_score = safe_int(row.get('credit_score'), 0)
        dti = safe_float(row.get('dti'), 0)
        ltv = safe_float(row.get('ltv'), 0)

        # Determine if repurchased
        repurchase_flag = str(row.get('repurchase_flag', 'N')).upper()
        zero_balance = str(row.get('zero_balance_code', ''))
        was_repurchased = repurchase_flag == 'Y' or zero_balance == '09'

        # Identify risk flags
        risk_flags = []
        if credit_score < 620:
            risk_flags.append("LOW_CREDIT_SCORE")
        if dti > 43:
            risk_flags.append("HIGH_DTI")
        if ltv > 95:
            risk_flags.append("HIGH_LTV")
        if was_repurchased:
            risk_flags.append("REPURCHASED")

        return FreddieLoan(
            loan_id=str(row.get('loan_sequence_number', '')),
            credit_score=credit_score,
            first_time_buyer=str(row.get('first_time_buyer', 'N')).upper() == 'Y',
            dti=dti,
            ltv=ltv,
            original_upb=safe_float(row.get('original_upb'), 0),
            interest_rate=safe_float(row.get('interest_rate'), 0),
            property_type=str(row.get('property_type', 'SF')),
            occupancy_status=str(row.get('occupancy_status', 'P')),
            property_state=str(row.get('property_state', '')),
            loan_purpose=str(row.get('loan_purpose', 'P')),
            num_borrowers=safe_int(row.get('num_borrowers'), 1),
            ever_delinquent=safe_int(row.get('current_delinquency'), 0) > 0,
            max_delinquency=safe_int(row.get('current_delinquency'), 0),
            zero_balance_code=zero_balance if zero_balance else None,
            was_repurchased=was_repurchased,
            was_modified=str(row.get('modification_flag', 'N')).upper() == 'Y',
            risk_flags=risk_flags
        )

    def get_repurchased_loans(self) -> List[FreddieLoan]:
        """Get all loans that were repurchased - key validation set."""
        loans = self.process_loans()
        return [l for l in loans if l.was_repurchased]

    def get_validation_split(self) -> Dict[str, List[FreddieLoan]]:
        """
        Split loans into validation categories.

        Returns:
            Dict with 'clean', 'repurchased', 'delinquent', 'modified' lists
        """
        loans = self.process_loans()

        return {
            'clean': [l for l in loans if not l.was_repurchased
                     and not l.ever_delinquent and not l.was_modified],
            'repurchased': [l for l in loans if l.was_repurchased],
            'delinquent': [l for l in loans if l.ever_delinquent and not l.was_repurchased],
            'modified': [l for l in loans if l.was_modified and not l.was_repurchased]
        }

    def generate_synthetic_documents(self, loan: FreddieLoan) -> Dict[str, Any]:
        """
        Generate synthetic loan documents from Freddie Mac characteristics.

        This creates realistic document content for testing our full pipeline
        without needing actual PII documents.
        """
        # Calculate implied income from DTI (rough approximation)
        # DTI = monthly_debt / monthly_income
        # Assume mortgage payment is ~80% of total debt
        rate = loan.interest_rate / 100 / 12 if loan.interest_rate > 0 else 0.005
        denom = 1 - (1 + rate) ** -360
        if abs(denom) < 1e-10:
            denom = 0.01  # Prevent division by zero
        estimated_payment = loan.original_upb * rate / denom
        estimated_monthly_debt = estimated_payment / 0.8 if estimated_payment > 0 else 5000
        estimated_monthly_income = estimated_monthly_debt / (loan.dti/100) if loan.dti > 0 else 10000
        estimated_annual_income = estimated_monthly_income * 12

        # Generate synthetic evidence statements
        evidence = {
            "loan_characteristics": {
                "credit_score": loan.credit_score,
                "dti": loan.dti,
                "ltv": loan.ltv,
                "loan_amount": loan.original_upb,
                "interest_rate": loan.interest_rate
            },
            "synthetic_documents": [],
            "evidence_statements": []
        }

        # W-2 / Income statement
        if loan.credit_score >= 620:
            evidence["synthetic_documents"].append({
                "type": "w2",
                "content": f"W-2 shows annual wages of ${estimated_annual_income:,.0f}"
            })
            evidence["evidence_statements"].append(
                f"Borrower's W-2 documents show annual income of ${estimated_annual_income:,.0f}."
            )

        # Credit report
        credit_tier = "excellent" if loan.credit_score >= 740 else \
                     "good" if loan.credit_score >= 680 else \
                     "fair" if loan.credit_score >= 620 else "poor"
        evidence["synthetic_documents"].append({
            "type": "credit_report",
            "content": f"Credit score: {loan.credit_score} ({credit_tier})"
        })
        evidence["evidence_statements"].append(
            f"Credit report shows score of {loan.credit_score}, rated as {credit_tier}."
        )

        # DTI analysis
        dti_status = "within guidelines" if loan.dti <= 43 else "exceeds standard guidelines"
        evidence["evidence_statements"].append(
            f"Debt-to-income ratio is {loan.dti}%, which {dti_status}."
        )

        # LTV analysis
        ltv_status = "standard" if loan.ltv <= 80 else \
                    "requires PMI" if loan.ltv <= 95 else "high risk"
        evidence["evidence_statements"].append(
            f"Loan-to-value ratio is {loan.ltv}%, classified as {ltv_status}."
        )

        # Add risk flags as evidence
        if loan.risk_flags:
            for flag in loan.risk_flags:
                if flag == "LOW_CREDIT_SCORE":
                    evidence["evidence_statements"].append(
                        "Credit score is below standard threshold of 620."
                    )
                elif flag == "HIGH_DTI":
                    evidence["evidence_statements"].append(
                        "DTI exceeds Freddie Mac standard maximum of 43%."
                    )
                elif flag == "HIGH_LTV":
                    evidence["evidence_statements"].append(
                        "LTV exceeds 95%, indicating high leverage risk."
                    )

        return evidence

    def summary(self) -> str:
        """Get summary of loaded data."""
        if self.orig_data is None:
            return "No data loaded"

        loans = self.process_loans()
        repurchased = [l for l in loans if l.was_repurchased]

        return f"""
FREDDIE MAC DATASET SUMMARY
===========================
Total Loans: {len(loans):,}
Repurchased: {len(repurchased):,} ({len(repurchased)/len(loans)*100:.2f}%)

Credit Score Distribution:
  < 620:  {sum(1 for l in loans if l.credit_score < 620):,}
  620-679: {sum(1 for l in loans if 620 <= l.credit_score < 680):,}
  680-739: {sum(1 for l in loans if 680 <= l.credit_score < 740):,}
  740+:   {sum(1 for l in loans if l.credit_score >= 740):,}

DTI Distribution:
  <= 43%: {sum(1 for l in loans if l.dti <= 43):,}
  > 43%:  {sum(1 for l in loans if l.dti > 43):,}

Repurchased Loan Characteristics:
  Avg Credit Score: {np.mean([l.credit_score for l in repurchased]) if repurchased else 0:.0f}
  Avg DTI: {np.mean([l.dti for l in repurchased]) if repurchased else 0:.1f}%
  Avg LTV: {np.mean([l.ltv for l in repurchased]) if repurchased else 0:.1f}%
"""
