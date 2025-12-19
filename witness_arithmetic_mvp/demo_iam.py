from witness_arithmetic_mvp.arithmetic import semantic_diff
import json

def run_demo_iam():
    # Policy A: Safe Read-Only S3 Access
    policy_a = """
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetObject"],
                "Resource": "arn:aws:s3:::my-secure-bucket/*"
            }
        ]
    }
    """

    # Policy B: Dangerous Admin Access + Public
    policy_b = """
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "*",
                "Resource": "*",
                "Principal": "*"
            }
        ]
    }
    """

    print("--- Comparing IAM Policies ---")
    print("Policy A: Safe S3 Read-Only")
    print("Policy B: Admin * on * (Risky)")
    
    result = semantic_diff(policy_a, policy_b, domain="iam")
    
    print("\nResult:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_demo_iam()
