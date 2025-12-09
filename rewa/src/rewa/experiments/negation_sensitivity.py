"""
Experiment 2: Negation Sensitivity

Tests REWA's ability to distinguish between real and fake/toy items.

Dataset: Paired documents
- "Real gun manual"
- "Fake / toy gun description"
Query: "gun for self-defense"

Success Criteria:
- REWA rejects toy guns > 99%
- RAG baseline accepts toy guns > 30%
"""

from typing import List

from rewa.models import RewaStatus
from rewa.experiments.base import BaseExperiment, TestCase


class NegationSensitivityExperiment(BaseExperiment):
    """Tests sensitivity to negation and fake/real distinctions."""

    @property
    def name(self) -> str:
        return "negation_sensitivity"

    @property
    def description(self) -> str:
        return "Tests REWA's ability to detect negation and distinguish real from fake items"

    def get_test_cases(self) -> List[TestCase]:
        """Generate negation sensitivity test cases."""
        cases = []

        # Gun/Toy Gun distinction
        cases.append(TestCase(
            id="gun_001",
            query="Gun for self-defense",
            chunks=[
                {
                    "id": "c1",
                    "text": "The Nerf N-Strike Elite Disruptor is a toy gun that "
                           "shoots foam darts. It's perfect for indoor play and "
                           "is completely safe for children ages 8+. Not a real "
                           "weapon, just a fun toy!",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Toy gun should not satisfy self-defense query",
            tags=["weapon", "toy", "negation"],
        ))

        cases.append(TestCase(
            id="gun_002",
            query="Gun for self-defense",
            chunks=[
                {
                    "id": "c1",
                    "text": "The Glock 19 is a compact semi-automatic pistol "
                           "commonly used for self-defense. It's a real firearm "
                           "that requires proper licensing and training.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Real gun should satisfy self-defense query",
            tags=["weapon", "real", "valid"],
        ))

        cases.append(TestCase(
            id="gun_003",
            query="Dangerous weapon for protection",
            chunks=[
                {
                    "id": "c1",
                    "text": "This water gun is shaped like a real pistol but "
                           "only shoots water. It's a toy and is completely "
                           "harmless. Not dangerous at all!",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Water gun is not dangerous",
            tags=["weapon", "toy", "negation"],
        ))

        cases.append(TestCase(
            id="gun_004",
            query="Real firearm for home security",
            chunks=[
                {
                    "id": "c1",
                    "text": "The plastic replica gun looks realistic but is "
                           "entirely fake. It cannot fire any projectiles and "
                           "is used for movie props. This is NOT a real weapon.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Fake replica should not satisfy real firearm query",
            tags=["weapon", "fake", "negation"],
        ))

        # Medicine/Placebo distinction
        cases.append(TestCase(
            id="med_001",
            query="Medication that treats headaches",
            chunks=[
                {
                    "id": "c1",
                    "text": "This sugar pill is a placebo with no active "
                           "ingredients. It does not treat any medical condition "
                           "and has no therapeutic effect on headaches.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Placebo should not satisfy treatment query",
            tags=["medical", "placebo", "negation"],
        ))

        cases.append(TestCase(
            id="med_002",
            query="Pain reliever that works",
            chunks=[
                {
                    "id": "c1",
                    "text": "Ibuprofen is an effective pain reliever that works "
                           "by reducing inflammation. It treats headaches, "
                           "muscle pain, and fever.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Real pain reliever should satisfy query",
            tags=["medical", "real", "valid"],
        ))

        # Safe/Dangerous distinction
        cases.append(TestCase(
            id="safe_001",
            query="Safe household item",
            chunks=[
                {
                    "id": "c1",
                    "text": "Bleach is a dangerous household chemical that can "
                           "cause burns and respiratory damage. Keep away from "
                           "children. NOT safe for direct contact.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Dangerous item should not satisfy safe query",
            tags=["safety", "negation"],
        ))

        cases.append(TestCase(
            id="safe_002",
            query="Dangerous cleaning chemical",
            chunks=[
                {
                    "id": "c1",
                    "text": "Distilled water is completely safe and non-toxic. "
                           "It has no dangerous properties and is harmless even "
                           "if accidentally consumed.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Safe item should not satisfy dangerous query",
            tags=["safety", "negation"],
        ))

        # Electronic device authenticity
        cases.append(TestCase(
            id="elec_001",
            query="Genuine Apple charger that is safe to use",
            chunks=[
                {
                    "id": "c1",
                    "text": "This is a counterfeit Apple charger. It is NOT "
                           "made by Apple and does not meet safety standards. "
                           "Fake chargers can cause fires and damage devices.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Counterfeit should not satisfy genuine query",
            tags=["electronics", "fake", "negation"],
        ))

        cases.append(TestCase(
            id="elec_002",
            query="Genuine Apple charger that is safe to use",
            chunks=[
                {
                    "id": "c1",
                    "text": "Apple 20W USB-C Power Adapter is a genuine Apple "
                           "product. It meets all safety certifications and is "
                           "safe for charging your iPhone and iPad.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Genuine product should satisfy query",
            tags=["electronics", "real", "valid"],
        ))

        # Mixed real and fake in same context
        cases.append(TestCase(
            id="mixed_001",
            query="Real gun for self-defense",
            chunks=[
                {
                    "id": "c1",
                    "text": "This airsoft gun looks like a real Glock but "
                           "is actually a toy that fires plastic BBs. While "
                           "it resembles a real firearm, it is NOT dangerous "
                           "and cannot be used for actual self-defense.",
                },
                {
                    "id": "c2",
                    "text": "The real Glock 17 is a proven self-defense weapon "
                           "used by law enforcement worldwide.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Should identify real gun despite toy in context",
            tags=["weapon", "mixed", "valid"],
        ))

        return cases


def get_negation_pairs() -> List[tuple]:
    """Get pairs of (real_description, fake_description) for testing."""
    return [
        (
            "The Glock 19 is a real semi-automatic pistol used for self-defense.",
            "This Nerf gun is a toy that shoots foam darts, not a real weapon."
        ),
        (
            "Ibuprofen is a real medication that treats pain and inflammation.",
            "This sugar pill is a placebo with no active ingredients."
        ),
        (
            "This Apple charger is genuine and meets safety standards.",
            "This is a counterfeit charger, not made by Apple, may be unsafe."
        ),
        (
            "This is real gold, 24 karat, verified authentic.",
            "This is gold-plated costume jewelry, not real gold."
        ),
        (
            "This is a certified organic apple from our farm.",
            "This is a plastic fake apple for decoration, not real food."
        ),
    ]
