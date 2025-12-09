"""
Experiment 4: Multi-Chart Ambiguity Stress Test

Tests REWA's ability to detect and handle semantic ambiguity.

Query: "Apple charger safety"
- Could be fruit pesticide (apple the fruit)
- Could be electronics overheating (Apple the company)

Success Criteria:
- Ambiguity detection > 90%
- System should ask clarifying question or preserve ambiguity
"""

from typing import List

from rewa.models import RewaStatus
from rewa.experiments.base import BaseExperiment, TestCase


class AmbiguityStressExperiment(BaseExperiment):
    """Tests multi-chart ambiguity detection capability."""

    @property
    def name(self) -> str:
        return "ambiguity_stress_test"

    @property
    def description(self) -> str:
        return "Tests REWA's ability to detect and handle semantic ambiguity"

    def get_test_cases(self) -> List[TestCase]:
        """Generate ambiguity stress test cases."""
        cases = []

        # Apple ambiguity (fruit vs company)
        cases.append(TestCase(
            id="apple_001",
            query="Apple charger safety",
            chunks=[
                {
                    "id": "c1",
                    "text": "Apple orchards use pesticides to protect fruit. "
                           "When charging organic matter, ensure proper "
                           "composting safety procedures. Apple fruit may "
                           "contain residual chemicals if not washed.",
                },
                {
                    "id": "c2",
                    "text": "Apple Inc. chargers should meet UL safety standards. "
                           "The USB-C charger can overheat if damaged. Only use "
                           "genuine Apple charging accessories to prevent fire.",
                },
            ],
            expected_status=RewaStatus.AMBIGUOUS,
            description="Should detect ambiguity between fruit and company",
            tags=["ambiguity", "apple"],
        ))

        # Bank ambiguity (financial vs river)
        cases.append(TestCase(
            id="bank_001",
            query="Bank safety measures",
            chunks=[
                {
                    "id": "c1",
                    "text": "River bank erosion can be dangerous. Safety measures "
                           "include installing barriers and planting vegetation "
                           "to stabilize the bank.",
                },
                {
                    "id": "c2",
                    "text": "Financial bank security requires vault protection, "
                           "armed guards, and surveillance systems to prevent "
                           "robbery and ensure customer safety.",
                },
            ],
            expected_status=RewaStatus.AMBIGUOUS,
            description="Should detect ambiguity between financial and river bank",
            tags=["ambiguity", "bank"],
        ))

        # Python ambiguity (snake vs programming)
        cases.append(TestCase(
            id="python_001",
            query="Python handling best practices",
            chunks=[
                {
                    "id": "c1",
                    "text": "Python snakes require careful handling. Always "
                           "support the body weight and avoid sudden movements. "
                           "Some species are venomous and dangerous.",
                },
                {
                    "id": "c2",
                    "text": "Python programming best practices include using "
                           "virtual environments, following PEP 8 style guide, "
                           "and handling exceptions properly.",
                },
            ],
            expected_status=RewaStatus.AMBIGUOUS,
            description="Should detect ambiguity between snake and programming",
            tags=["ambiguity", "python"],
        ))

        # Mercury ambiguity (planet vs element)
        cases.append(TestCase(
            id="mercury_001",
            query="Mercury exposure risks",
            chunks=[
                {
                    "id": "c1",
                    "text": "Mercury the planet has extreme temperature variations. "
                           "Exposure to solar radiation is intense on the surface. "
                           "Space suits provide protection from cosmic rays.",
                },
                {
                    "id": "c2",
                    "text": "Mercury poisoning from the heavy metal can cause "
                           "neurological damage. Exposure risks include broken "
                           "thermometers and contaminated fish consumption.",
                },
            ],
            expected_status=RewaStatus.AMBIGUOUS,
            description="Should detect ambiguity between planet and element",
            tags=["ambiguity", "mercury"],
        ))

        # Jaguar ambiguity (animal vs car)
        cases.append(TestCase(
            id="jaguar_001",
            query="Jaguar maintenance requirements",
            chunks=[
                {
                    "id": "c1",
                    "text": "Jaguars in captivity require specialized care. "
                           "Their enclosures need regular maintenance including "
                           "cleaning, temperature control, and enrichment.",
                },
                {
                    "id": "c2",
                    "text": "Jaguar automobiles require premium maintenance. "
                           "Oil changes every 10,000 miles, brake inspections, "
                           "and annual servicing at authorized dealers.",
                },
            ],
            expected_status=RewaStatus.AMBIGUOUS,
            description="Should detect ambiguity between animal and car",
            tags=["ambiguity", "jaguar"],
        ))

        # Non-ambiguous cases (should NOT be flagged)
        cases.append(TestCase(
            id="clear_001",
            query="iPhone battery replacement",
            chunks=[
                {
                    "id": "c1",
                    "text": "iPhone battery replacement can be done at Apple "
                           "stores or authorized service providers. Costs vary "
                           "by model. DIY replacement voids warranty.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Clear context should not be ambiguous",
            tags=["non-ambiguous", "valid"],
        ))

        cases.append(TestCase(
            id="clear_002",
            query="Python 3 installation guide",
            chunks=[
                {
                    "id": "c1",
                    "text": "To install Python 3, download from python.org. "
                           "On Windows, use the installer. On Mac, use brew "
                           "install python3. Verify with python3 --version.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Specific context (Python 3) removes ambiguity",
            tags=["non-ambiguous", "valid"],
        ))

        cases.append(TestCase(
            id="clear_003",
            query="Wells Fargo bank account",
            chunks=[
                {
                    "id": "c1",
                    "text": "Wells Fargo offers checking and savings accounts. "
                           "Online banking is available 24/7. Minimum balance "
                           "requirements vary by account type.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Specific bank name removes ambiguity",
            tags=["non-ambiguous", "valid"],
        ))

        # Multiple domains with clear resolution
        cases.append(TestCase(
            id="multi_001",
            query="Apple iPhone safety recalls",
            chunks=[
                {
                    "id": "c1",
                    "text": "Apple has issued safety recalls for certain iPhone "
                           "battery models. Affected devices may overheat. "
                           "Check Apple's website for your serial number.",
                },
                {
                    "id": "c2",
                    "text": "Apple fruit recalls due to listeria contamination "
                           "affect certain growers. Check FDA website for "
                           "affected batches.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Specific term 'iPhone' should resolve to electronics",
            tags=["disambiguation", "valid"],
        ))

        return cases


def get_ambiguous_terms() -> List[dict]:
    """Get list of ambiguous terms with their multiple meanings."""
    return [
        {
            "term": "Apple",
            "meanings": ["Fruit", "Technology company"],
            "queries": ["Apple nutrition", "Apple products", "Apple safety"],
        },
        {
            "term": "Bank",
            "meanings": ["Financial institution", "River bank", "Blood bank"],
            "queries": ["Bank deposits", "Bank erosion", "Bank storage"],
        },
        {
            "term": "Python",
            "meanings": ["Snake", "Programming language"],
            "queries": ["Python syntax", "Python habitat", "Python security"],
        },
        {
            "term": "Mercury",
            "meanings": ["Planet", "Chemical element", "Roman god"],
            "queries": ["Mercury temperature", "Mercury poisoning", "Mercury symbolism"],
        },
        {
            "term": "Java",
            "meanings": ["Programming language", "Island", "Coffee"],
            "queries": ["Java installation", "Java tourism", "Java brewing"],
        },
        {
            "term": "Virus",
            "meanings": ["Biological pathogen", "Computer malware"],
            "queries": ["Virus protection", "Virus symptoms", "Virus removal"],
        },
        {
            "term": "Shell",
            "meanings": ["Seashell", "Oil company", "Unix shell"],
            "queries": ["Shell commands", "Shell collection", "Shell station"],
        },
        {
            "term": "Eclipse",
            "meanings": ["Astronomical event", "IDE", "Car model"],
            "queries": ["Eclipse timing", "Eclipse installation", "Eclipse specs"],
        },
    ]
