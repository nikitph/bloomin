"""
Policy Corpus: Freddie Mac Seller Guide

Version-locked policy loading with clause classification:
- Mandatory clauses (MUST)
- Conditional clauses (MAY/IF)
- Explicit exclusions (NOT PERMITTED)

Focus: Sections 5301-5303 (Income Stability & Acceptability)
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ClauseType(Enum):
    """Types of policy clauses."""
    MANDATORY = "mandatory"      # MUST, SHALL, REQUIRED
    CONDITIONAL = "conditional"  # MAY, IF, WHEN
    EXCLUSION = "exclusion"      # NOT PERMITTED, PROHIBITED
    GUIDANCE = "guidance"        # SHOULD, RECOMMENDED


@dataclass
class PolicyClause:
    """A single policy clause from the Freddie Mac Guide."""
    id: str
    section: str
    subsection: str
    text: str
    clause_type: ClauseType
    keywords: List[str] = field(default_factory=list)
    effective_date: str = ""
    supersedes: Optional[str] = None


@dataclass
class PolicyVersion:
    """A version-locked policy document."""
    version_id: str
    effective_date: str
    source: str
    hash: str
    clauses: List[PolicyClause]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FreddieMacPolicyCorpus:
    """
    Freddie Mac Seller Guide policy corpus.

    Version-locked for auditability.
    Focused on Income sections 5301-5303.
    """

    def __init__(self):
        self.current_version: Optional[PolicyVersion] = None
        self.version_history: List[str] = []

    def load_income_policies(self) -> PolicyVersion:
        """
        Load Freddie Mac income-related policies.

        Based on Single-Family Seller/Servicer Guide sections 5301-5303.
        This is a curated subset focused on income stability.
        """
        clauses = []

        # =================================================================
        # SECTION 5301: GENERAL INCOME REQUIREMENTS
        # =================================================================

        clauses.append(PolicyClause(
            id="5301.1",
            section="5301",
            subsection="General Requirements",
            text="The Seller must determine that the Borrower has a reasonable expectation of continued income sufficient to support the mortgage payment.",
            clause_type=ClauseType.MANDATORY,
            keywords=["income", "continued", "sufficient", "mortgage payment"]
        ))

        clauses.append(PolicyClause(
            id="5301.2",
            section="5301",
            subsection="Income Documentation",
            text="All income used for qualifying must be documented and verified. Verbal verification alone is not sufficient.",
            clause_type=ClauseType.MANDATORY,
            keywords=["documented", "verified", "verbal verification"]
        ))

        clauses.append(PolicyClause(
            id="5301.3",
            section="5301",
            subsection="Income Stability",
            text="Income must demonstrate stability. A minimum of two years of documented income history is required unless specific exceptions apply.",
            clause_type=ClauseType.MANDATORY,
            keywords=["stability", "two years", "history"]
        ))

        clauses.append(PolicyClause(
            id="5301.4",
            section="5301",
            subsection="Declining Income",
            text="If income shows a declining trend, the Seller must use the lower income figure or provide documented justification for using a higher amount.",
            clause_type=ClauseType.MANDATORY,
            keywords=["declining", "trend", "lower income", "justification"]
        ))

        # =================================================================
        # SECTION 5302: EMPLOYMENT INCOME
        # =================================================================

        clauses.append(PolicyClause(
            id="5302.1",
            section="5302",
            subsection="Base Employment Income",
            text="Base employment income may be used when the Borrower has been employed in the same line of work for at least two years.",
            clause_type=ClauseType.CONDITIONAL,
            keywords=["base income", "employed", "two years", "same line of work"]
        ))

        clauses.append(PolicyClause(
            id="5302.2",
            section="5302",
            subsection="Employment Gaps",
            text="Employment gaps greater than 30 days within the most recent two years must be explained and documented.",
            clause_type=ClauseType.MANDATORY,
            keywords=["employment gaps", "30 days", "explained", "documented"]
        ))

        clauses.append(PolicyClause(
            id="5302.3",
            section="5302",
            subsection="Variable Income - Commission",
            text="Commission income may be used when the Borrower has received commission income for at least two years and it demonstrates a pattern of stability or increase.",
            clause_type=ClauseType.CONDITIONAL,
            keywords=["commission", "two years", "stability", "increase"]
        ))

        clauses.append(PolicyClause(
            id="5302.4",
            section="5302",
            subsection="Variable Income - Bonus",
            text="Bonus income may be used when the Borrower has received bonus income for at least two years. Declining bonus income requires the use of the lower average.",
            clause_type=ClauseType.CONDITIONAL,
            keywords=["bonus", "two years", "declining", "lower average"]
        ))

        clauses.append(PolicyClause(
            id="5302.5",
            section="5302",
            subsection="Part-Time Income",
            text="Part-time income may be used if the Borrower has worked part-time for at least two years and the income has been consistent.",
            clause_type=ClauseType.CONDITIONAL,
            keywords=["part-time", "two years", "consistent"]
        ))

        clauses.append(PolicyClause(
            id="5302.6",
            section="5302",
            subsection="Recent Employment Change",
            text="If a Borrower has changed jobs within the past two years but remained in the same line of work with equal or greater income, current income may be used.",
            clause_type=ClauseType.CONDITIONAL,
            keywords=["changed jobs", "same line of work", "equal or greater"]
        ))

        # =================================================================
        # SECTION 5303: SELF-EMPLOYMENT INCOME
        # =================================================================

        clauses.append(PolicyClause(
            id="5303.1",
            section="5303",
            subsection="Self-Employment Definition",
            text="A Borrower is considered self-employed if they have 25% or greater ownership interest in a business.",
            clause_type=ClauseType.MANDATORY,
            keywords=["self-employed", "25%", "ownership"]
        ))

        clauses.append(PolicyClause(
            id="5303.2",
            section="5303",
            subsection="Self-Employment History",
            text="Self-employment income requires a minimum of two years of documented self-employment history in the same business.",
            clause_type=ClauseType.MANDATORY,
            keywords=["self-employment", "two years", "same business"]
        ))

        clauses.append(PolicyClause(
            id="5303.3",
            section="5303",
            subsection="Self-Employment Documentation",
            text="Self-employment income must be documented with signed tax returns including all schedules for the most recent two years.",
            clause_type=ClauseType.MANDATORY,
            keywords=["tax returns", "schedules", "two years"]
        ))

        clauses.append(PolicyClause(
            id="5303.4",
            section="5303",
            subsection="Self-Employment Declining Income",
            text="If self-employment income has declined by 20% or more year-over-year, the Seller must obtain additional documentation demonstrating the decline is not expected to continue.",
            clause_type=ClauseType.MANDATORY,
            keywords=["declined", "20%", "year-over-year", "additional documentation"]
        ))

        clauses.append(PolicyClause(
            id="5303.5",
            section="5303",
            subsection="Business Viability",
            text="The Seller must verify that the business is currently operating and has adequate liquidity to support ongoing operations.",
            clause_type=ClauseType.MANDATORY,
            keywords=["business", "operating", "liquidity"]
        ))

        # =================================================================
        # SECTION 5304: GIG / CONTRACT INCOME (NEW ADDITIONS)
        # =================================================================

        clauses.append(PolicyClause(
            id="5304.1",
            section="5304",
            subsection="Gig Income Definition",
            text="Gig or contract income includes earnings from short-term contracts, freelance work, or platform-based employment where the Borrower is not a W-2 employee.",
            clause_type=ClauseType.GUIDANCE,
            keywords=["gig", "contract", "freelance", "platform"]
        ))

        clauses.append(PolicyClause(
            id="5304.2",
            section="5304",
            subsection="Gig Income Requirements",
            text="Gig income may only be used when documented for at least two years with consistent or increasing earnings patterns.",
            clause_type=ClauseType.CONDITIONAL,
            keywords=["gig", "two years", "consistent", "increasing"]
        ))

        clauses.append(PolicyClause(
            id="5304.3",
            section="5304",
            subsection="Contract Income Documentation",
            text="Contract income must be verified with copies of contracts, 1099 forms, and bank statements showing deposits.",
            clause_type=ClauseType.MANDATORY,
            keywords=["contract", "1099", "bank statements"]
        ))

        # =================================================================
        # EXCLUSIONS AND PROHIBITIONS
        # =================================================================

        clauses.append(PolicyClause(
            id="EX.1",
            section="Exclusions",
            subsection="Unverifiable Income",
            text="Income that cannot be verified through documentation is not permitted for qualification purposes.",
            clause_type=ClauseType.EXCLUSION,
            keywords=["unverifiable", "not permitted"]
        ))

        clauses.append(PolicyClause(
            id="EX.2",
            section="Exclusions",
            subsection="Projected Income",
            text="Projected or anticipated future income that has not yet been received is not permitted unless the Borrower has a guaranteed employment contract.",
            clause_type=ClauseType.EXCLUSION,
            keywords=["projected", "anticipated", "future income", "not permitted"]
        ))

        clauses.append(PolicyClause(
            id="EX.3",
            section="Exclusions",
            subsection="One-Time Income",
            text="One-time or non-recurring income sources (lottery, inheritance, legal settlements) are not permitted for qualifying income.",
            clause_type=ClauseType.EXCLUSION,
            keywords=["one-time", "non-recurring", "lottery", "inheritance"]
        ))

        clauses.append(PolicyClause(
            id="EX.4",
            section="Exclusions",
            subsection="Income Without History",
            text="New income sources without at least 12 months of documented history are not permitted unless offset by significant compensating factors.",
            clause_type=ClauseType.EXCLUSION,
            keywords=["new income", "12 months", "not permitted"]
        ))

        # Create version
        version_text = json.dumps([c.__dict__ for c in clauses], default=str)
        version_hash = hashlib.sha256(version_text.encode()).hexdigest()[:16]

        version = PolicyVersion(
            version_id=f"FM-2024-{version_hash[:8]}",
            effective_date="2024-01-01",
            source="Freddie Mac Single-Family Seller/Servicer Guide (Curated)",
            hash=version_hash,
            clauses=clauses,
            metadata={
                "sections": ["5301", "5302", "5303", "5304", "Exclusions"],
                "focus": "Income Stability & Acceptability",
                "total_clauses": len(clauses),
                "mandatory_count": sum(1 for c in clauses if c.clause_type == ClauseType.MANDATORY),
                "conditional_count": sum(1 for c in clauses if c.clause_type == ClauseType.CONDITIONAL),
                "exclusion_count": sum(1 for c in clauses if c.clause_type == ClauseType.EXCLUSION)
            }
        )

        self.current_version = version
        self.version_history.append(version.version_id)

        return version

    def get_clauses_by_section(self, section: str) -> List[PolicyClause]:
        """Get all clauses for a specific section."""
        if not self.current_version:
            return []
        return [c for c in self.current_version.clauses if c.section == section]

    def get_clauses_by_type(self, clause_type: ClauseType) -> List[PolicyClause]:
        """Get all clauses of a specific type."""
        if not self.current_version:
            return []
        return [c for c in self.current_version.clauses if c.clause_type == clause_type]

    def get_mandatory_clauses(self) -> List[PolicyClause]:
        """Get all mandatory (MUST) clauses."""
        return self.get_clauses_by_type(ClauseType.MANDATORY)

    def get_exclusions(self) -> List[PolicyClause]:
        """Get all exclusion (NOT PERMITTED) clauses."""
        return self.get_clauses_by_type(ClauseType.EXCLUSION)

    def search_clauses(self, keywords: List[str]) -> List[PolicyClause]:
        """Search clauses by keywords."""
        if not self.current_version:
            return []

        results = []
        for clause in self.current_version.clauses:
            # Check if any keyword matches
            clause_text_lower = clause.text.lower()
            clause_keywords_lower = [k.lower() for k in clause.keywords]

            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in clause_text_lower or keyword_lower in clause_keywords_lower:
                    results.append(clause)
                    break

        return results

    def get_policy_embedding_texts(self) -> List[str]:
        """Get all clause texts for embedding."""
        if not self.current_version:
            return []
        return [c.text for c in self.current_version.clauses]

    def verify_version_integrity(self) -> bool:
        """Verify the policy version hasn't been tampered with."""
        if not self.current_version:
            return False

        version_text = json.dumps(
            [c.__dict__ for c in self.current_version.clauses],
            default=str
        )
        current_hash = hashlib.sha256(version_text.encode()).hexdigest()[:16]

        return current_hash == self.current_version.hash

    def export_for_audit(self, filepath: str):
        """Export policy version for audit trail."""
        if not self.current_version:
            raise ValueError("No policy loaded")

        export_data = {
            "version_id": self.current_version.version_id,
            "effective_date": self.current_version.effective_date,
            "source": self.current_version.source,
            "hash": self.current_version.hash,
            "export_timestamp": datetime.now().isoformat(),
            "metadata": self.current_version.metadata,
            "clauses": [
                {
                    "id": c.id,
                    "section": c.section,
                    "subsection": c.subsection,
                    "text": c.text,
                    "type": c.clause_type.value,
                    "keywords": c.keywords
                }
                for c in self.current_version.clauses
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    def summary(self) -> str:
        """Get a summary of the loaded policy."""
        if not self.current_version:
            return "No policy loaded"

        return f"""
FREDDIE MAC POLICY CORPUS
========================
Version: {self.current_version.version_id}
Effective: {self.current_version.effective_date}
Hash: {self.current_version.hash}

Clauses by Type:
  Mandatory (MUST): {self.current_version.metadata['mandatory_count']}
  Conditional (MAY): {self.current_version.metadata['conditional_count']}
  Exclusions (NOT PERMITTED): {self.current_version.metadata['exclusion_count']}

Sections: {', '.join(self.current_version.metadata['sections'])}
Focus: {self.current_version.metadata['focus']}
"""
