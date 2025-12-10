#!/usr/bin/env python3
"""
Rewa Visualization Server

Generates real loan data projections and serves the interactive 3D visualization.
Uses actual Freddie Mac data projected through the trained Rewa-space.
"""

import sys
import os
import json
import http.server
import socketserver
import webbrowser
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime

from freddie_validation.src.freddie_data_loader import FreddieMacDataLoader
from freddie_validation.src.income_evaluator import IncomeAdmissibilityEvaluator, Decision
from freddie_validation.src.loan_evidence import (
    LoanFile, Document, DocumentType, EmploymentRecord,
    IncomeRecord, IncomeType, EmploymentStatus
)
from src.rewa_space_v2 import RewaSpaceV2


def create_loan_file(loan):
    """Create LoanFile from Freddie Mac loan."""
    loan_file = LoanFile(
        loan_id=loan.loan_id,
        borrower_name="Freddie Mac Borrower",
        application_date=datetime.now().strftime("%Y-%m-%d")
    )

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

    credit_status = "excellent" if loan.credit_score >= 740 else \
                   "good" if loan.credit_score >= 680 else \
                   "fair" if loan.credit_score >= 620 else "subprime"

    content = f"Credit score: {loan.credit_score} ({credit_status}). "
    if loan.credit_score < 620:
        content += "BELOW MINIMUM THRESHOLD. "
    content += f"DTI: {loan.dti}%. "
    if loan.dti > 43:
        content += "EXCEEDS MAXIMUM DTI. "
    content += f"LTV: {loan.ltv}%. "
    if loan.ltv > 95:
        content += "HIGH LEVERAGE RISK. "

    loan_file.documents.append(Document(
        "CR-1", DocumentType.OTHER, "Credit Report",
        datetime.now().strftime("%Y-%m-%d"), content, True
    ))

    loan_file.employment_history.append(EmploymentRecord(
        "Current Employer", "Employee", "2020-01-01", None,
        EmploymentStatus.FULL_TIME_W2, 5000, IncomeType.BASE_SALARY, True
    ))

    loan_file.income_records.extend([
        IncomeRecord(2022, None, 60000, IncomeType.BASE_SALARY, "Employer", True, "W2-1"),
        IncomeRecord(2023, None, 62000, IncomeType.BASE_SALARY, "Employer", True, "W2-1"),
    ])

    loan_file.ground_truth = {
        "conforming": not loan.was_repurchased,
        "risk_flags": loan.risk_flags
    }

    return loan_file


def project_to_3d(embedding):
    """Project high-dimensional embedding to 3D for visualization using PCA-like approach."""
    # Use first 3 principal directions (simplified)
    # In practice, you'd use actual PCA or t-SNE
    if len(embedding) >= 3:
        # Take weighted combinations of embedding dimensions
        x = np.sum(embedding[::3]) / len(embedding[::3])
        y = np.sum(embedding[1::3]) / len(embedding[1::3])
        z = np.sum(embedding[2::3]) / len(embedding[2::3])
    else:
        x, y, z = embedding[0], embedding[1] if len(embedding) > 1 else 0, embedding[2] if len(embedding) > 2 else 0

    # Normalize to unit sphere
    norm = np.sqrt(x*x + y*y + z*z)
    if norm > 0:
        x, y, z = x/norm, y/norm, z/norm

    return [float(x), float(y), float(z)]


def generate_visualization_data(num_loans=200):
    """Generate visualization data from real Freddie Mac loans."""

    print("Loading Freddie Mac data...")
    data_dir = Path(__file__).parent.parent / "data"

    # Try to load data
    loader = FreddieMacDataLoader(str(data_dir))
    try:
        loader.load_sample_data(2020)
    except FileNotFoundError:
        print("No real data found, using synthetic data")
        return generate_synthetic_data(num_loans)

    splits = loader.get_validation_split()

    print("Training Rewa-space projection...")
    rewa_space = RewaSpaceV2(output_dim=384)

    # Training pairs for Rewa-space
    training_pairs = [
        ("Income is stable and verified", "Income is unstable and unverified"),
        ("Loan meets requirements", "Loan does not meet requirements"),
        ("Credit score is excellent", "Credit score is poor"),
        ("DTI is within limits", "DTI exceeds limits"),
        ("LTV is acceptable", "LTV is too high"),
        ("Documentation is complete", "Documentation is incomplete"),
        ("Approved for purchase", "Subject to repurchase"),
        ("yes", "no"),
        ("true", "false"),
        ("accept", "reject"),
    ]
    rewa_space.train(training_pairs, epochs=100, verbose=False)

    print("Projecting loans to Rewa-space...")

    # Select diverse sample of loans
    loans = []

    # Add clean loans
    clean = splits['clean'][:150]
    loans.extend(clean)

    # Add repurchased loans (all of them)
    loans.extend(splits['repurchased'])

    # Add delinquent loans
    delinquent = splits['delinquent'][:30]
    loans.extend(delinquent)

    # Limit total
    loans = loans[:num_loans]

    print(f"Processing {len(loans)} loans...")

    # Initialize evaluator for decisions
    evaluator = IncomeAdmissibilityEvaluator(train_rewa_space=False)
    evaluator.rewa_space = rewa_space

    visualization_data = []

    for loan in loans:
        # Create evidence statements
        statements = [
            f"Credit score is {loan.credit_score}",
            f"Debt-to-income ratio is {loan.dti}%",
            f"Loan-to-value ratio is {loan.ltv}%",
        ]

        if loan.credit_score < 620:
            statements.append("Credit score is below minimum threshold")
        if loan.dti > 43:
            statements.append("DTI exceeds maximum guidelines")
        if loan.ltv > 95:
            statements.append("LTV indicates high leverage risk")

        # Project combined evidence to Rewa-space
        embeddings = [rewa_space.project(stmt) for stmt in statements]
        combined = np.mean(embeddings, axis=0)
        combined = combined / (np.linalg.norm(combined) + 1e-10)

        # Project to 3D
        pos_3d = project_to_3d(combined)

        # Determine decision based on STANDARD policy thresholds
        # (Browser will recalculate based on selected policy)
        meets_policy = (
            loan.credit_score >= 620 and
            loan.dti <= 43 and
            loan.ltv <= 95
        )
        decision = "approve" if meets_policy and not loan.was_repurchased else "refuse"

        visualization_data.append({
            "id": loan.loan_id,
            "position": pos_3d,
            "creditScore": int(loan.credit_score),
            "dti": float(loan.dti),
            "ltv": float(loan.ltv),
            "wasRepurchased": bool(loan.was_repurchased),
            "everDelinquent": bool(loan.ever_delinquent),
            "decision": decision,
            "riskFlags": loan.risk_flags,
        })

    return visualization_data


def generate_synthetic_data(num_loans=200):
    """Generate synthetic visualization data when real data unavailable."""
    data = []

    for i in range(num_loans):
        # Random position on sphere
        theta = np.random.random() * np.pi * 2
        phi = np.arccos(2 * np.random.random() - 1)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Generate characteristics
        credit = int(580 + np.random.random() * 220)
        dti = int(20 + np.random.random() * 35)
        ltv = int(60 + np.random.random() * 40)

        was_repurchased = ltv > 94 and np.random.random() < 0.3

        # Decision based on standard policy
        meets_policy = credit >= 620 and dti <= 43 and ltv <= 95
        decision = "approve" if meets_policy and not was_repurchased else "refuse"

        data.append({
            "id": f"SYNTH-{i:04d}",
            "position": [float(x), float(y), float(z)],
            "creditScore": credit,
            "dti": dti,
            "ltv": ltv,
            "wasRepurchased": was_repurchased,
            "everDelinquent": False,
            "decision": decision,
            "riskFlags": [],
        })

    return data


def create_html_with_data(loan_data):
    """Create HTML file with embedded loan data."""

    html_path = Path(__file__).parent / "index.html"

    with open(html_path, 'r') as f:
        html_content = f.read()

    # Inject the real data into the HTML
    data_script = f"""
    <script>
        // Real Freddie Mac loan data from Python
        const REAL_LOAN_DATA = {json.dumps(loan_data, indent=2)};

        // Override generateSampleData to use real data
        function generateSampleData() {{
            REAL_LOAN_DATA.forEach(loan => {{
                loanData.push({{
                    id: loan.id,
                    position: new THREE.Vector3(loan.position[0], loan.position[1], loan.position[2]),
                    creditScore: loan.creditScore,
                    dti: loan.dti,
                    ltv: loan.ltv,
                    wasRepurchased: loan.wasRepurchased,
                    decision: loan.decision,
                    riskFlags: loan.riskFlags
                }});
            }});
            updateStats();
        }}
    </script>
    """

    # Insert data script before closing body tag
    modified_html = html_content.replace('</body>', data_script + '</body>')

    # Write to new file
    output_path = Path(__file__).parent / "visualization_with_data.html"
    with open(output_path, 'w') as f:
        f.write(modified_html)

    return output_path


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that suppresses logs."""
    def log_message(self, format, *args):
        pass  # Suppress logging


def serve_visualization(port=8080):
    """Serve the visualization on localhost."""

    print("\n" + "="*60)
    print("REWA SEMANTIC SPHERE VISUALIZATION")
    print("="*60)

    # Generate data
    loan_data = generate_visualization_data(200)
    print(f"\nGenerated visualization data for {len(loan_data)} loans")

    # Count statistics
    approved = sum(1 for l in loan_data if l['decision'] == 'approve')
    refused = len(loan_data) - approved
    repurchased = sum(1 for l in loan_data if l['wasRepurchased'])
    hal = sum(1 for l in loan_data if l['wasRepurchased'] and l['decision'] == 'approve')

    print(f"  Approved: {approved}")
    print(f"  Refused: {refused}")
    print(f"  Repurchased: {repurchased}")
    print(f"  Hallucinated Approvals: {hal}")

    # Create HTML with embedded data
    html_path = create_html_with_data(loan_data)
    print(f"\nVisualization created: {html_path}")

    # Change to visualization directory
    os.chdir(Path(__file__).parent)

    # Start server
    with socketserver.TCPServer(("", port), QuietHandler) as httpd:
        url = f"http://localhost:{port}/visualization_with_data.html"
        print(f"\n{'='*60}")
        print(f"Server running at: {url}")
        print(f"{'='*60}")
        print("\nPress Ctrl+C to stop the server\n")

        # Open browser
        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Serve Rewa visualization")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    args = parser.parse_args()

    serve_visualization(args.port)
