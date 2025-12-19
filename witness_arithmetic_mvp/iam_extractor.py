import json

def extract_witnesses_iam(policy_json: str) -> set[str]:
    """
    Extracts semantic witnesses from an IAM Policy JSON.
    """
    witnesses = set()
    try:
        policy = json.loads(policy_json)
    except json.JSONDecodeError:
        print("Warning: Invalid JSON for IAM policy")
        return witnesses

    statements = policy.get("Statement", [])
    if isinstance(statements, dict):
        statements = [statements]

    for stmt in statements:
        effect = stmt.get("Effect", "")
        actions = stmt.get("Action", [])
        resources = stmt.get("Resource", [])
        principal = stmt.get("Principal", "")
        
        if isinstance(actions, str): actions = [actions]
        if isinstance(resources, str): resources = [resources]

        # Effects
        if effect == "Allow": witnesses.add("allow")
        if effect == "Deny": witnesses.add("deny")
        
        # Actions
        for action in actions:
            if action == "*":
                witnesses.add("admin_all")
            elif action.startswith("s3:List"): witnesses.add("s3_list")
            elif action.startswith("s3:Get"): witnesses.add("s3_read")
            elif action.startswith("s3:Put"): witnesses.add("s3_write")
            elif action.startswith("s3:Delete"): witnesses.add("s3_delete")
            elif action.startswith("ec2:Run"): witnesses.add("ec2_run")
            elif action.startswith("ec2:Term"): witnesses.add("ec2_term")
            elif action.startswith("ec2:Describe"): witnesses.add("ec2_describe")
            
        # Resources
        for res in resources:
            if res == "*":
                witnesses.add("resource_wildcard")
            else:
                if "bucket" in res: witnesses.add("specific_bucket")
                
        # Risk Patterns
        # Admin Privilege: Allow * on *
        if effect == "Allow" and "*" in actions and "*" in resources:
            witnesses.add("admin_privilege")
            
        # Public Access: Principal *
        if principal == "*" or (isinstance(principal, dict) and principal.get("AWS") == "*"):
            witnesses.add("public_access")
            
    return witnesses
