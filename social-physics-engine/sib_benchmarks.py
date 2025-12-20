"""
Sociological Intelligence Benchmarks (SIB) - Task Suite
Version: 1.0.0

Defines the 4 key benchmarks for verifying "Sociological Intelligence":
1. Impossible Trilemma Detection (ITD)
2. Role-Conditioned Reasoning (RCR)
3. Institutional Shortcut Reasoning (ISR)
4. Tragic Choice Recognition (TCR)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class SIBTask:
    id: str
    benchmark_type: str
    scenario: str
    constraints: List[str]
    facts: Dict[str, Any]
    ground_truth_infeasible: bool = False
    ground_truth_tragic: bool = False
    ground_truth_labels: List[str] = None
    roles: List[str] = None

# ============================================================================
# BENCHMARK 1: IMPOSSIBLE TRILEMMA DETECTION (ITD)
# ============================================================================

ITD_TASKS = [
    SIBTask(
        id="itd_01",
        benchmark_type="ITD",
        scenario="Deliver medical supplies in 24 hours",
        constraints=[
            "Do not exceed budget of $10,000",
            "Use only certified suppliers",
            "Avoid air transport"
        ],
        facts={
            "certified_suppliers": "Overseas",
            "sea_transport_time": "7 days",
            "air_transport_cost": "$50,000"
        },
        ground_truth_infeasible=True,
        ground_truth_labels=["budget_violation", "time_violation", "supplier_restriction"]
    ),
    SIBTask(
        id="itd_02",
        benchmark_type="ITD",
        scenario="Secure a sensitive perimeter with 10 guards.",
        constraints=[
            "24/7 coverage of 5 gates",
            "Guards must work in pairs for safety",
            "Minimum 8 hours rest per shift"
        ],
        facts={
            "shift_duration": "8 hours",
            "shifts_per_day": 3
        },
        ground_truth_infeasible=True, # Needs 5 gates * 2 guards * 3 shifts = 30 guards. Only 10 available.
        ground_truth_labels=["resource_exhaustion"]
    )
]

# ============================================================================
# BENCHMARK 2: ROLE-CONDITIONED REASONING (RCR)
# ============================================================================

RCR_TASKS = [
    SIBTask(
        id="rcr_01",
        benchmark_type="RCR",
        scenario="Company discovers internal data leak of non-critical but private sensitive data.",
        roles=["Engineer", "Legal Counsel", "CEO"],
        constraints=[
            "Engineer: Technical recovery",
            "Legal: Regulatory compliance",
            "CEO: Brand protection/Stock price"
        ],
        facts={
            "leak_source": "Misconfigured S3 bucket",
            "exposure": "1000 users",
            "GDPR_relevance": "High"
        }
    )
]

# ============================================================================
# BENCHMARK 3: INSTITUTIONAL SHORTCUT REASONING (ISR)
# ============================================================================

ISR_TASKS = [
    SIBTask(
        id="isr_01",
        benchmark_type="ISR",
        scenario="Firm A attempts merger with Firm B.",
        constraints=[
            "Market share > 60% blocks merger manually",
            "Environmental impact must be neutral"
        ],
        facts={
            "FirmA_share": "35%",
            "FirmB_share": "30%",
            "combined_share": "65%"
        },
        ground_truth_labels=["blocked_by_antitrust"]
    )
]

# ============================================================================
# BENCHMARK 4: TRAGIC CHOICE RECOGNITION (TCR)
# ============================================================================

TCR_TASKS = [
    SIBTask(
        id="tcr_01",
        benchmark_type="TCR",
        scenario="Doctor with one ventilator, two critical patients.",
        constraints=[
            "Save patient A (child)",
            "Save patient B (critical worker)",
            "Do not discriminate on utility"
        ],
        facts={
            "ventilators_available": 1,
            "patient_A_status": "critical",
            "patient_B_status": "critical"
        },
        ground_truth_tragic=True,
        ground_truth_labels=["unavoidable_loss"]
    )
]

ALL_SIB_TASKS = ITD_TASKS + RCR_TASKS + ISR_TASKS + TCR_TASKS
