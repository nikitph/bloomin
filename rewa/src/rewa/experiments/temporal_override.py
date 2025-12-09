"""
Experiment 3: Temporal Override (Medical)

Tests REWA's ability to handle temporal information in medical records.

Dataset: Synthetic EHR timelines with allergies changing over time

Success Criteria:
- Temporal correctness > 95%
"""

from datetime import datetime, timedelta
from typing import List

from rewa.models import RewaStatus
from rewa.experiments.base import BaseExperiment, TestCase


class TemporalOverrideExperiment(BaseExperiment):
    """Tests temporal reasoning with medical records."""

    @property
    def name(self) -> str:
        return "temporal_override"

    @property
    def description(self) -> str:
        return "Tests REWA's ability to handle temporal changes in medical information"

    def get_test_cases(self) -> List[TestCase]:
        """Generate temporal override test cases."""
        cases = []

        # Allergy changes over time
        cases.append(TestCase(
            id="allergy_001",
            query="What is the patient's current allergy status for penicillin?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Medical Record - 2020-01-15: Patient John Doe has "
                           "a documented allergy to penicillin. Reaction: severe "
                           "rash and difficulty breathing.",
                    "metadata": {"date": "2020-01-15"},
                },
                {
                    "id": "c2",
                    "text": "Medical Record - 2023-06-20: After allergy testing, "
                           "patient John Doe's penicillin allergy has been "
                           "removed from record. No allergic reaction detected "
                           "in controlled testing. Patient is NOT allergic to "
                           "penicillin as of this date.",
                    "metadata": {"date": "2023-06-20"},
                },
            ],
            expected_status=RewaStatus.VALID,
            expected_properties={"confidence": 0.5},  # Should use latest
            description="Latest record should override earlier allergy status",
            tags=["medical", "temporal", "allergy"],
        ))

        cases.append(TestCase(
            id="allergy_002",
            query="Can the patient take aspirin?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Medical Record - 2019-03-10: Patient Jane Smith "
                           "can safely take aspirin. No known allergies.",
                    "metadata": {"date": "2019-03-10"},
                },
                {
                    "id": "c2",
                    "text": "Medical Record - 2022-08-15: Patient Jane Smith "
                           "developed aspirin sensitivity. DO NOT PRESCRIBE "
                           "aspirin. Patient is now allergic.",
                    "metadata": {"date": "2022-08-15"},
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Later allergy development should override earlier safety",
            tags=["medical", "temporal", "allergy"],
        ))

        # Medication changes
        cases.append(TestCase(
            id="med_001",
            query="What is the patient's current blood pressure medication?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Prescription - 2021-01-05: Patient Bob Johnson "
                           "prescribed Lisinopril 10mg daily for hypertension.",
                    "metadata": {"date": "2021-01-05"},
                },
                {
                    "id": "c2",
                    "text": "Prescription - 2021-06-15: Lisinopril discontinued "
                           "due to cough. Patient Bob Johnson now on Losartan "
                           "50mg daily instead.",
                    "metadata": {"date": "2021-06-15"},
                },
                {
                    "id": "c3",
                    "text": "Prescription - 2023-02-20: Dosage adjustment. "
                           "Patient Bob Johnson Losartan increased to 100mg daily.",
                    "metadata": {"date": "2023-02-20"},
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Should return most recent medication",
            tags=["medical", "temporal", "medication"],
        ))

        # Diagnosis changes
        cases.append(TestCase(
            id="diag_001",
            query="Does the patient have diabetes?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Lab Results - 2020-03-15: Blood glucose test for "
                           "patient Mary Lee. Fasting glucose: 85 mg/dL. "
                           "No diabetes detected.",
                    "metadata": {"date": "2020-03-15"},
                },
                {
                    "id": "c2",
                    "text": "Lab Results - 2023-09-10: Blood glucose test for "
                           "patient Mary Lee. Fasting glucose: 145 mg/dL. "
                           "A1C: 7.2%. Diagnosis: Type 2 Diabetes.",
                    "metadata": {"date": "2023-09-10"},
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Latest diagnosis should take precedence",
            tags=["medical", "temporal", "diagnosis"],
        ))

        cases.append(TestCase(
            id="diag_002",
            query="Is the patient in remission from cancer?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Oncology Report - 2021-04-20: Patient Tom White "
                           "diagnosed with Stage 2 breast cancer. Beginning "
                           "chemotherapy treatment.",
                    "metadata": {"date": "2021-04-20"},
                },
                {
                    "id": "c2",
                    "text": "Oncology Report - 2023-01-15: Patient Tom White "
                           "completed treatment. CT scan shows no evidence of "
                           "disease. Patient is in complete remission.",
                    "metadata": {"date": "2023-01-15"},
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Remission status should override earlier diagnosis",
            tags=["medical", "temporal", "cancer"],
        ))

        # Conflicting records from same time period
        cases.append(TestCase(
            id="conflict_001",
            query="What is the patient's blood type?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Lab Report - 2022-05-10: Patient ID 12345. "
                           "Blood type: A positive. Verified.",
                    "metadata": {"date": "2022-05-10"},
                },
                {
                    "id": "c2",
                    "text": "Lab Report - 2022-05-10: Patient ID 12345. "
                           "Blood type: O negative. Verified.",
                    "metadata": {"date": "2022-05-10"},
                },
            ],
            expected_status=RewaStatus.CONFLICT,
            description="Same-day conflicting records should be flagged",
            tags=["medical", "temporal", "conflict"],
        ))

        # Weight/BMI tracking
        cases.append(TestCase(
            id="vital_001",
            query="What is the patient's current weight?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Vitals - 2022-01-10: Weight: 180 lbs",
                    "metadata": {"date": "2022-01-10"},
                },
                {
                    "id": "c2",
                    "text": "Vitals - 2022-06-15: Weight: 175 lbs",
                    "metadata": {"date": "2022-06-15"},
                },
                {
                    "id": "c3",
                    "text": "Vitals - 2023-03-20: Weight: 165 lbs",
                    "metadata": {"date": "2023-03-20"},
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Should return most recent weight",
            tags=["medical", "temporal", "vitals"],
        ))

        return cases


def generate_synthetic_ehr(
    patient_id: str,
    num_records: int = 10,
    start_date: datetime = None
) -> List[dict]:
    """
    Generate synthetic EHR timeline for testing.

    Args:
        patient_id: Patient identifier
        num_records: Number of records to generate
        start_date: Starting date for timeline

    Returns:
        List of synthetic medical records
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365 * 3)

    records = []
    current_date = start_date

    allergies = ["penicillin", "none"]
    medications = ["Lisinopril", "Losartan", "Metformin"]

    for i in range(num_records):
        record_type = i % 3

        if record_type == 0:
            # Allergy record
            allergy = allergies[i % len(allergies)]
            text = f"Medical Record - {current_date.strftime('%Y-%m-%d')}: " \
                   f"Patient {patient_id} allergy status: {allergy}"
            if allergy == "none":
                text += ". No known allergies."
            else:
                text += f". Allergic to {allergy}."
        elif record_type == 1:
            # Medication record
            med = medications[i % len(medications)]
            dose = (i + 1) * 10
            text = f"Prescription - {current_date.strftime('%Y-%m-%d')}: " \
                   f"Patient {patient_id} prescribed {med} {dose}mg daily."
        else:
            # Vitals record
            weight = 180 - (i * 2)
            text = f"Vitals - {current_date.strftime('%Y-%m-%d')}: " \
                   f"Patient {patient_id} Weight: {weight} lbs"

        records.append({
            "id": f"rec_{patient_id}_{i}",
            "text": text,
            "metadata": {
                "date": current_date.strftime("%Y-%m-%d"),
                "patient_id": patient_id,
                "record_type": ["allergy", "medication", "vitals"][record_type],
            },
        })

        # Advance date randomly
        current_date += timedelta(days=30 + (i * 7))

    return records
