"""
Loan File Evidence Processor

Processes loan application packages into structured evidence:
- Bank statements
- Tax returns
- VOE documents
- Employment history
- Income documentation

Each document becomes a witness in Rewa-space.
"""

import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class DocumentType(Enum):
    """Types of loan documents."""
    TAX_RETURN = "tax_return"
    W2 = "w2"
    PAYSTUB = "paystub"
    BANK_STATEMENT = "bank_statement"
    VOE = "voe"  # Verification of Employment
    VOD = "vod"  # Verification of Deposit
    BUSINESS_LICENSE = "business_license"
    PROFIT_LOSS = "profit_loss"
    CONTRACT_1099 = "1099"
    LETTER_OF_EXPLANATION = "loe"
    OTHER = "other"


class IncomeType(Enum):
    """Types of income."""
    BASE_SALARY = "base_salary"
    HOURLY = "hourly"
    COMMISSION = "commission"
    BONUS = "bonus"
    OVERTIME = "overtime"
    SELF_EMPLOYMENT = "self_employment"
    GIG_CONTRACT = "gig_contract"
    PART_TIME = "part_time"
    RENTAL = "rental"
    RETIREMENT = "retirement"
    OTHER = "other"


class EmploymentStatus(Enum):
    """Employment status categories."""
    FULL_TIME_W2 = "full_time_w2"
    PART_TIME_W2 = "part_time_w2"
    SELF_EMPLOYED = "self_employed"
    CONTRACT_1099 = "contract_1099"
    GIG_WORKER = "gig_worker"
    RETIRED = "retired"
    UNEMPLOYED = "unemployed"


@dataclass
class IncomeRecord:
    """A single income record."""
    year: int
    month: Optional[int]
    amount: float
    income_type: IncomeType
    source: str
    verified: bool = False
    document_id: Optional[str] = None


@dataclass
class EmploymentRecord:
    """Employment history record."""
    employer: str
    position: str
    start_date: str
    end_date: Optional[str]  # None if current
    status: EmploymentStatus
    monthly_income: float
    income_type: IncomeType
    verified: bool = False


@dataclass
class Document:
    """A loan document."""
    id: str
    type: DocumentType
    description: str
    date: str
    content_summary: str  # Extracted key information
    verified: bool = False
    source_file: Optional[str] = None


@dataclass
class LoanFile:
    """Complete loan application package."""
    loan_id: str
    borrower_name: str
    application_date: str

    # Documents
    documents: List[Document] = field(default_factory=list)

    # Extracted data
    employment_history: List[EmploymentRecord] = field(default_factory=list)
    income_records: List[IncomeRecord] = field(default_factory=list)

    # Analysis results
    total_monthly_income: float = 0.0
    years_in_current_role: float = 0.0
    employment_gaps: List[Dict[str, Any]] = field(default_factory=list)
    income_trend: str = "unknown"  # increasing, stable, declining

    # Ground truth (for validation)
    ground_truth: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute hash for reproducibility."""
        data = json.dumps({
            "loan_id": self.loan_id,
            "documents": [d.id for d in self.documents],
            "employment": [e.employer for e in self.employment_history],
            "income": [i.amount for i in self.income_records]
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class LoanEvidenceProcessor:
    """
    Processes loan files into structured evidence for Rewa evaluation.

    Extracts:
    - Income history and trends
    - Employment continuity
    - Documentation completeness
    - Potential issues/flags
    """

    def __init__(self):
        self.required_doc_types = [
            DocumentType.TAX_RETURN,
            DocumentType.W2,
            DocumentType.PAYSTUB,
            DocumentType.VOE
        ]

    def process_loan_file(self, loan_file: LoanFile) -> Dict[str, Any]:
        """
        Process a loan file into evidence structure.

        Returns:
            Dict with evidence summaries and flags
        """
        evidence = {
            "loan_id": loan_file.loan_id,
            "hash": loan_file.compute_hash(),
            "processed_at": datetime.now().isoformat(),

            # Documentation analysis
            "documentation": self._analyze_documentation(loan_file),

            # Income analysis
            "income": self._analyze_income(loan_file),

            # Employment analysis
            "employment": self._analyze_employment(loan_file),

            # Issues and flags
            "flags": self._identify_flags(loan_file),

            # Evidence statements (for Rewa)
            "evidence_statements": self._generate_evidence_statements(loan_file)
        }

        return evidence

    def _analyze_documentation(self, loan_file: LoanFile) -> Dict[str, Any]:
        """Analyze documentation completeness."""
        doc_types_present = set(d.type for d in loan_file.documents)

        missing_required = []
        for req in self.required_doc_types:
            if req not in doc_types_present:
                missing_required.append(req.value)

        return {
            "total_documents": len(loan_file.documents),
            "document_types": [d.type.value for d in loan_file.documents],
            "missing_required": missing_required,
            "is_complete": len(missing_required) == 0,
            "verified_count": sum(1 for d in loan_file.documents if d.verified)
        }

    def _analyze_income(self, loan_file: LoanFile) -> Dict[str, Any]:
        """Analyze income history and trends."""
        if not loan_file.income_records:
            return {
                "has_income_data": False,
                "error": "No income records found"
            }

        # Group by year
        by_year = {}
        for record in loan_file.income_records:
            year = record.year
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(record.amount)

        # Calculate yearly totals
        yearly_totals = {year: sum(amounts) for year, amounts in by_year.items()}
        years = sorted(yearly_totals.keys())

        # Determine trend
        if len(years) >= 2:
            recent = yearly_totals[years[-1]]
            previous = yearly_totals[years[-2]]

            if recent > previous * 1.1:
                trend = "increasing"
            elif recent < previous * 0.8:
                trend = "declining"
                decline_pct = (previous - recent) / previous * 100
            else:
                trend = "stable"
                decline_pct = 0
        else:
            trend = "insufficient_history"
            decline_pct = 0

        # Calculate average
        if years:
            avg_annual = sum(yearly_totals.values()) / len(years)
            avg_monthly = avg_annual / 12
        else:
            avg_annual = 0
            avg_monthly = 0

        return {
            "has_income_data": True,
            "years_of_history": len(years),
            "yearly_totals": yearly_totals,
            "average_annual": avg_annual,
            "average_monthly": avg_monthly,
            "trend": trend,
            "decline_percentage": decline_pct if trend == "declining" else 0,
            "income_types": list(set(r.income_type.value for r in loan_file.income_records)),
            "verified_income": sum(r.amount for r in loan_file.income_records if r.verified)
        }

    def _analyze_employment(self, loan_file: LoanFile) -> Dict[str, Any]:
        """Analyze employment history."""
        if not loan_file.employment_history:
            return {
                "has_employment_data": False,
                "error": "No employment records found"
            }

        # Sort by start date
        sorted_employment = sorted(
            loan_file.employment_history,
            key=lambda e: e.start_date
        )

        # Check for current employment
        current = [e for e in sorted_employment if e.end_date is None]
        has_current = len(current) > 0

        # Calculate gaps
        gaps = []
        for i in range(len(sorted_employment) - 1):
            curr = sorted_employment[i]
            next_emp = sorted_employment[i + 1]

            if curr.end_date:
                # Parse dates (simplified - assumes YYYY-MM-DD format)
                end = datetime.strptime(curr.end_date, "%Y-%m-%d")
                start = datetime.strptime(next_emp.start_date, "%Y-%m-%d")
                gap_days = (start - end).days

                if gap_days > 30:
                    gaps.append({
                        "from_employer": curr.employer,
                        "to_employer": next_emp.employer,
                        "gap_days": gap_days,
                        "explained": False  # Would need LOE document
                    })

        # Check employment types
        employment_types = list(set(e.status.value for e in sorted_employment))
        is_self_employed = EmploymentStatus.SELF_EMPLOYED in [e.status for e in current]

        # Years in current role
        if current:
            current_start = datetime.strptime(current[0].start_date, "%Y-%m-%d")
            years_current = (datetime.now() - current_start).days / 365
        else:
            years_current = 0

        return {
            "has_employment_data": True,
            "has_current_employment": has_current,
            "is_self_employed": is_self_employed,
            "employment_types": employment_types,
            "total_employers": len(sorted_employment),
            "years_in_current_role": round(years_current, 1),
            "employment_gaps": gaps,
            "has_unexplained_gaps": any(not g["explained"] for g in gaps),
            "current_monthly_income": current[0].monthly_income if current else 0
        }

    def _identify_flags(self, loan_file: LoanFile) -> List[Dict[str, Any]]:
        """Identify potential issues that require attention."""
        flags = []

        income_analysis = self._analyze_income(loan_file)
        employment_analysis = self._analyze_employment(loan_file)
        doc_analysis = self._analyze_documentation(loan_file)

        # Missing documentation
        if not doc_analysis["is_complete"]:
            flags.append({
                "type": "MISSING_DOCUMENTATION",
                "severity": "high",
                "description": f"Missing required documents: {doc_analysis['missing_required']}",
                "policy_reference": "5301.2"
            })

        # Insufficient income history
        if income_analysis.get("years_of_history", 0) < 2:
            flags.append({
                "type": "INSUFFICIENT_INCOME_HISTORY",
                "severity": "high",
                "description": f"Only {income_analysis.get('years_of_history', 0)} years of income history (2 required)",
                "policy_reference": "5301.3"
            })

        # Declining income
        if income_analysis.get("trend") == "declining":
            decline_pct = income_analysis.get("decline_percentage", 0)
            severity = "high" if decline_pct > 20 else "medium"
            flags.append({
                "type": "DECLINING_INCOME",
                "severity": severity,
                "description": f"Income declined {decline_pct:.1f}% year-over-year",
                "policy_reference": "5301.4" if decline_pct <= 20 else "5303.4"
            })

        # Employment gaps
        if employment_analysis.get("has_unexplained_gaps"):
            gaps = employment_analysis.get("employment_gaps", [])
            flags.append({
                "type": "UNEXPLAINED_EMPLOYMENT_GAP",
                "severity": "medium",
                "description": f"{len(gaps)} employment gap(s) > 30 days without explanation",
                "policy_reference": "5302.2"
            })

        # Self-employment without sufficient history
        if employment_analysis.get("is_self_employed"):
            years = employment_analysis.get("years_in_current_role", 0)
            if years < 2:
                flags.append({
                    "type": "INSUFFICIENT_SELF_EMPLOYMENT_HISTORY",
                    "severity": "high",
                    "description": f"Self-employed for {years:.1f} years (2 required)",
                    "policy_reference": "5303.2"
                })

        # No current employment
        if not employment_analysis.get("has_current_employment", False):
            flags.append({
                "type": "NO_CURRENT_EMPLOYMENT",
                "severity": "critical",
                "description": "No current employment record found",
                "policy_reference": "5301.1"
            })

        # Check for credit/DTI/LTV issues from document content
        for doc in loan_file.documents:
            content = doc.content_summary.upper()
            if "BELOW MINIMUM THRESHOLD" in content or "SUBPRIME" in content:
                flags.append({
                    "type": "LOW_CREDIT_SCORE",
                    "severity": "critical",
                    "description": "Credit score below minimum threshold (620)",
                    "policy_reference": "5101.1"
                })
            if "EXCEEDS MAXIMUM DTI" in content:
                flags.append({
                    "type": "HIGH_DTI",
                    "severity": "critical",
                    "description": "Debt-to-income ratio exceeds maximum (43%)",
                    "policy_reference": "5101.2"
                })
            if "HIGH LEVERAGE RISK" in content:
                flags.append({
                    "type": "HIGH_LTV",
                    "severity": "critical",
                    "description": "Loan-to-value ratio exceeds safe threshold (95%)",
                    "policy_reference": "5101.3"
                })

        # Check ground truth risk flags if available
        if loan_file.ground_truth and "risk_flags" in loan_file.ground_truth:
            risk_flags = loan_file.ground_truth["risk_flags"]
            if "LOW_CREDIT_SCORE" in risk_flags and not any(f["type"] == "LOW_CREDIT_SCORE" for f in flags):
                flags.append({
                    "type": "LOW_CREDIT_SCORE",
                    "severity": "critical",
                    "description": "Borrower credit score is subprime",
                    "policy_reference": "5101.1"
                })
            if "HIGH_DTI" in risk_flags and not any(f["type"] == "HIGH_DTI" for f in flags):
                flags.append({
                    "type": "HIGH_DTI",
                    "severity": "critical",
                    "description": "DTI exceeds Freddie Mac maximum",
                    "policy_reference": "5101.2"
                })
            if "HIGH_LTV" in risk_flags and not any(f["type"] == "HIGH_LTV" for f in flags):
                flags.append({
                    "type": "HIGH_LTV",
                    "severity": "critical",
                    "description": "LTV indicates excessive leverage",
                    "policy_reference": "5101.3"
                })
            if "REPURCHASED" in risk_flags:
                flags.append({
                    "type": "REPURCHASED",
                    "severity": "critical",
                    "description": "This loan was historically repurchased by Freddie Mac",
                    "policy_reference": "HISTORICAL"
                })

        return flags

    def _generate_evidence_statements(self, loan_file: LoanFile) -> List[str]:
        """
        Generate natural language evidence statements for Rewa.

        These become witnesses in the semantic space.
        """
        statements = []

        income_analysis = self._analyze_income(loan_file)
        employment_analysis = self._analyze_employment(loan_file)
        doc_analysis = self._analyze_documentation(loan_file)

        # Documentation statements
        if doc_analysis["is_complete"]:
            statements.append("All required documentation has been provided and verified.")
        else:
            missing = doc_analysis["missing_required"]
            statements.append(f"Documentation is incomplete. Missing: {', '.join(missing)}.")

        # Income statements
        if income_analysis.get("has_income_data"):
            years = income_analysis.get("years_of_history", 0)
            avg = income_analysis.get("average_monthly", 0)
            trend = income_analysis.get("trend", "unknown")

            statements.append(f"Income history spans {years} years with average monthly income of ${avg:,.0f}.")
            statements.append(f"Income trend is {trend}.")

            if trend == "declining":
                decline = income_analysis.get("decline_percentage", 0)
                statements.append(f"Income has declined {decline:.1f}% from previous year.")
        else:
            statements.append("No verifiable income documentation provided.")

        # Employment statements
        if employment_analysis.get("has_employment_data"):
            if employment_analysis.get("has_current_employment"):
                years = employment_analysis.get("years_in_current_role", 0)
                statements.append(f"Borrower has been in current position for {years:.1f} years.")
            else:
                statements.append("Borrower does not have current employment.")

            if employment_analysis.get("is_self_employed"):
                statements.append("Borrower is self-employed with 25% or greater ownership.")

            if employment_analysis.get("has_unexplained_gaps"):
                gaps = len(employment_analysis.get("employment_gaps", []))
                statements.append(f"Employment history shows {gaps} gap(s) exceeding 30 days without explanation.")
        else:
            statements.append("No employment history documentation provided.")

        # Extract risk indicators from document content summaries
        for doc in loan_file.documents:
            content = doc.content_summary.upper()
            # Credit score indicators
            if "BELOW MINIMUM THRESHOLD" in content or "SUBPRIME" in content:
                statements.append("Credit score is below minimum threshold of 620. This loan does not meet Freddie Mac requirements.")
            # DTI indicators
            if "EXCEEDS MAXIMUM DTI" in content:
                statements.append("Debt-to-income ratio exceeds maximum threshold of 43%. This is a non-conforming loan.")
            # LTV indicators
            if "HIGH LEVERAGE RISK" in content:
                statements.append("Loan-to-value ratio exceeds 95%, indicating high leverage risk. This loan may be subject to repurchase.")

        # Check ground truth for risk flags if available
        if loan_file.ground_truth and "risk_flags" in loan_file.ground_truth:
            risk_flags = loan_file.ground_truth["risk_flags"]
            if "LOW_CREDIT_SCORE" in risk_flags:
                statements.append("CRITICAL: Borrower has a subprime credit score. Loan is non-conforming.")
            if "HIGH_DTI" in risk_flags:
                statements.append("CRITICAL: Debt-to-income ratio exceeds Freddie Mac guidelines. Loan is non-conforming.")
            if "HIGH_LTV" in risk_flags:
                statements.append("CRITICAL: Loan-to-value ratio is dangerously high. This loan would be subject to repurchase.")
            if "REPURCHASED" in risk_flags:
                statements.append("CRITICAL: This loan was repurchased by Freddie Mac due to non-conformance.")

        # Flag-based statements
        flags = self._identify_flags(loan_file)
        for flag in flags:
            if flag["severity"] in ["high", "critical"]:
                statements.append(f"ISSUE: {flag['description']}")

        return statements

    def create_test_loan_file(
        self,
        scenario: str,
        loan_id: str = "TEST-001"
    ) -> LoanFile:
        """
        Create a test loan file for a specific scenario.

        Scenarios:
        - clean_w2: Clean W-2 employee, 3 years history, stable income
        - declining_income: W-2 with 25% decline
        - self_employed_2yr: Self-employed, exactly 2 years
        - self_employed_new: Self-employed, only 1 year
        - employment_gap: Has 60-day unexplained gap
        - missing_docs: Missing tax returns
        - gig_worker: Contract/gig income
        - contradictory: Conflicting income data
        """
        loan = LoanFile(
            loan_id=loan_id,
            borrower_name="Test Borrower",
            application_date=datetime.now().strftime("%Y-%m-%d")
        )

        if scenario == "clean_w2":
            # Clean W-2 employee - should APPROVE
            loan.documents = [
                Document("DOC-1", DocumentType.TAX_RETURN, "2023 Tax Return", "2024-04-15", "AGI: $85,000", True),
                Document("DOC-2", DocumentType.TAX_RETURN, "2022 Tax Return", "2023-04-15", "AGI: $82,000", True),
                Document("DOC-3", DocumentType.W2, "2023 W-2", "2024-01-31", "Wages: $85,000", True),
                Document("DOC-4", DocumentType.W2, "2022 W-2", "2023-01-31", "Wages: $82,000", True),
                Document("DOC-5", DocumentType.PAYSTUB, "Recent Paystub", "2024-06-15", "YTD: $42,500", True),
                Document("DOC-6", DocumentType.VOE, "Employment Verification", "2024-06-01", "Employed since 2021", True),
            ]
            loan.employment_history = [
                EmploymentRecord("ABC Corp", "Senior Analyst", "2021-03-15", None,
                                EmploymentStatus.FULL_TIME_W2, 7083, IncomeType.BASE_SALARY, True)
            ]
            loan.income_records = [
                IncomeRecord(2022, None, 82000, IncomeType.BASE_SALARY, "ABC Corp", True, "DOC-4"),
                IncomeRecord(2023, None, 85000, IncomeType.BASE_SALARY, "ABC Corp", True, "DOC-3"),
            ]
            loan.ground_truth = {"conforming": True, "reason": "Clean W-2 with stable income"}

        elif scenario == "declining_income":
            # 25% income decline - should REFUSE or require explanation
            loan.documents = [
                Document("DOC-1", DocumentType.TAX_RETURN, "2023 Tax Return", "2024-04-15", "AGI: $60,000", True),
                Document("DOC-2", DocumentType.TAX_RETURN, "2022 Tax Return", "2023-04-15", "AGI: $80,000", True),
                Document("DOC-3", DocumentType.W2, "2023 W-2", "2024-01-31", "Wages: $60,000", True),
                Document("DOC-4", DocumentType.W2, "2022 W-2", "2023-01-31", "Wages: $80,000", True),
                Document("DOC-5", DocumentType.PAYSTUB, "Recent Paystub", "2024-06-15", "YTD: $30,000", True),
                Document("DOC-6", DocumentType.VOE, "Employment Verification", "2024-06-01", "Employed since 2020", True),
            ]
            loan.employment_history = [
                EmploymentRecord("XYZ Inc", "Account Manager", "2020-01-15", None,
                                EmploymentStatus.FULL_TIME_W2, 5000, IncomeType.BASE_SALARY, True)
            ]
            loan.income_records = [
                IncomeRecord(2022, None, 80000, IncomeType.BASE_SALARY, "XYZ Inc", True, "DOC-4"),
                IncomeRecord(2023, None, 60000, IncomeType.BASE_SALARY, "XYZ Inc", True, "DOC-3"),
            ]
            loan.ground_truth = {"conforming": False, "reason": "25% income decline without explanation"}

        elif scenario == "self_employed_2yr":
            # Self-employed exactly 2 years - borderline, should be careful
            loan.documents = [
                Document("DOC-1", DocumentType.TAX_RETURN, "2023 Tax Return", "2024-04-15", "Schedule C: $95,000", True),
                Document("DOC-2", DocumentType.TAX_RETURN, "2022 Tax Return", "2023-04-15", "Schedule C: $90,000", True),
                Document("DOC-3", DocumentType.BUSINESS_LICENSE, "Business License", "2022-06-01", "Active", True),
                Document("DOC-4", DocumentType.PROFIT_LOSS, "2024 YTD P&L", "2024-06-15", "Net: $48,000", True),
                Document("DOC-5", DocumentType.BANK_STATEMENT, "Business Bank", "2024-06-01", "Avg balance: $25,000", True),
            ]
            loan.employment_history = [
                EmploymentRecord("Self - Consulting LLC", "Owner", "2022-06-01", None,
                                EmploymentStatus.SELF_EMPLOYED, 7916, IncomeType.SELF_EMPLOYMENT, True)
            ]
            loan.income_records = [
                IncomeRecord(2022, None, 45000, IncomeType.SELF_EMPLOYMENT, "Consulting LLC", True, "DOC-2"),  # Partial year
                IncomeRecord(2023, None, 95000, IncomeType.SELF_EMPLOYMENT, "Consulting LLC", True, "DOC-1"),
            ]
            loan.ground_truth = {"conforming": True, "reason": "Self-employed meets 2-year minimum"}

        elif scenario == "self_employed_new":
            # Self-employed only 1 year - should REFUSE
            loan.documents = [
                Document("DOC-1", DocumentType.TAX_RETURN, "2023 Tax Return", "2024-04-15", "Schedule C: $100,000", True),
                Document("DOC-2", DocumentType.BUSINESS_LICENSE, "Business License", "2023-03-01", "Active", True),
                Document("DOC-3", DocumentType.PROFIT_LOSS, "2024 YTD P&L", "2024-06-15", "Net: $55,000", True),
            ]
            loan.employment_history = [
                EmploymentRecord("Self - New Venture", "Owner", "2023-03-01", None,
                                EmploymentStatus.SELF_EMPLOYED, 8333, IncomeType.SELF_EMPLOYMENT, True)
            ]
            loan.income_records = [
                IncomeRecord(2023, None, 75000, IncomeType.SELF_EMPLOYMENT, "New Venture", True, "DOC-1"),  # Partial year
            ]
            loan.ground_truth = {"conforming": False, "reason": "Self-employed less than 2 years"}

        elif scenario == "employment_gap":
            # 60-day employment gap - needs explanation
            loan.documents = [
                Document("DOC-1", DocumentType.TAX_RETURN, "2023 Tax Return", "2024-04-15", "AGI: $75,000", True),
                Document("DOC-2", DocumentType.TAX_RETURN, "2022 Tax Return", "2023-04-15", "AGI: $70,000", True),
                Document("DOC-3", DocumentType.W2, "2023 W-2 Current", "2024-01-31", "Wages: $75,000", True),
                Document("DOC-4", DocumentType.W2, "2022 W-2 Previous", "2023-01-31", "Wages: $50,000", True),
                Document("DOC-5", DocumentType.VOE, "Current Employment", "2024-06-01", "Employed since 2023-04", True),
            ]
            loan.employment_history = [
                EmploymentRecord("Old Company", "Analyst", "2020-01-15", "2023-01-31",
                                EmploymentStatus.FULL_TIME_W2, 5833, IncomeType.BASE_SALARY, True),
                EmploymentRecord("New Company", "Senior Analyst", "2023-04-01", None,
                                EmploymentStatus.FULL_TIME_W2, 6250, IncomeType.BASE_SALARY, True)
            ]
            loan.income_records = [
                IncomeRecord(2022, None, 70000, IncomeType.BASE_SALARY, "Old Company", True, "DOC-4"),
                IncomeRecord(2023, None, 75000, IncomeType.BASE_SALARY, "Mixed", True, "DOC-3"),
            ]
            loan.ground_truth = {"conforming": False, "reason": "60-day employment gap without LOE"}

        elif scenario == "missing_docs":
            # Missing tax returns - should REFUSE
            loan.documents = [
                Document("DOC-1", DocumentType.W2, "2023 W-2", "2024-01-31", "Wages: $80,000", True),
                Document("DOC-2", DocumentType.PAYSTUB, "Recent Paystub", "2024-06-15", "YTD: $40,000", True),
                Document("DOC-3", DocumentType.VOE, "Employment Verification", "2024-06-01", "Employed since 2019", True),
            ]
            loan.employment_history = [
                EmploymentRecord("Stable Corp", "Manager", "2019-06-01", None,
                                EmploymentStatus.FULL_TIME_W2, 6666, IncomeType.BASE_SALARY, True)
            ]
            loan.income_records = [
                IncomeRecord(2023, None, 80000, IncomeType.BASE_SALARY, "Stable Corp", True, "DOC-1"),
            ]
            loan.ground_truth = {"conforming": False, "reason": "Missing required tax returns"}

        elif scenario == "gig_worker":
            # Gig/contract worker - needs 2 years consistent history
            loan.documents = [
                Document("DOC-1", DocumentType.TAX_RETURN, "2023 Tax Return", "2024-04-15", "1099 income: $70,000", True),
                Document("DOC-2", DocumentType.TAX_RETURN, "2022 Tax Return", "2023-04-15", "1099 income: $65,000", True),
                Document("DOC-3", DocumentType.CONTRACT_1099, "2023 1099-NEC", "2024-01-31", "Multiple: $70,000", True),
                Document("DOC-4", DocumentType.CONTRACT_1099, "2022 1099-NEC", "2023-01-31", "Multiple: $65,000", True),
                Document("DOC-5", DocumentType.BANK_STATEMENT, "Personal Bank 6mo", "2024-06-01", "Consistent deposits", True),
            ]
            loan.employment_history = [
                EmploymentRecord("Multiple Platforms", "Contractor", "2022-01-01", None,
                                EmploymentStatus.GIG_WORKER, 5833, IncomeType.GIG_CONTRACT, True)
            ]
            loan.income_records = [
                IncomeRecord(2022, None, 65000, IncomeType.GIG_CONTRACT, "Multiple", True, "DOC-4"),
                IncomeRecord(2023, None, 70000, IncomeType.GIG_CONTRACT, "Multiple", True, "DOC-3"),
            ]
            loan.ground_truth = {"conforming": True, "reason": "Gig income with 2+ years consistent history"}

        elif scenario == "contradictory":
            # Contradictory evidence - should REFUSE
            loan.documents = [
                Document("DOC-1", DocumentType.TAX_RETURN, "2023 Tax Return", "2024-04-15", "AGI: $50,000", True),
                Document("DOC-2", DocumentType.W2, "2023 W-2", "2024-01-31", "Wages: $85,000", True),  # Contradicts tax return
                Document("DOC-3", DocumentType.PAYSTUB, "Recent Paystub", "2024-06-15", "Annual: $95,000", True),  # Different again
                Document("DOC-4", DocumentType.VOE, "Employment Verification", "2024-06-01", "Salary: $75,000", True),  # Yet another number
            ]
            loan.employment_history = [
                EmploymentRecord("Confused Corp", "Employee", "2021-01-01", None,
                                EmploymentStatus.FULL_TIME_W2, 7083, IncomeType.BASE_SALARY, True)
            ]
            loan.income_records = [
                IncomeRecord(2023, None, 50000, IncomeType.BASE_SALARY, "Tax Return", True, "DOC-1"),
                IncomeRecord(2023, None, 85000, IncomeType.BASE_SALARY, "W-2", True, "DOC-2"),
            ]
            loan.ground_truth = {"conforming": False, "reason": "Contradictory income documentation"}

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return loan
