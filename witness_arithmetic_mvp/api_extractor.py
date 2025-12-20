import yaml
import json

def extract_witnesses_api(schema_str: str) -> set[str]:
    """
    Extracts semantic witnesses from OpenAPI/Swagger schemas (JSON or YAML).
    """
    witnesses = set()
    try:
        # Try JSON first, then YAML
        spec = json.loads(schema_str)
    except json.JSONDecodeError:
        try:
            spec = yaml.safe_load(schema_str)
        except yaml.YAMLError:
            print("Warning: Invalid API Schema")
            return witnesses

    if not spec: return witnesses

    paths = spec.get("paths", {})
    
    for path, methods in paths.items():
        for method, details in methods.items():
            method = method.lower()
            if method in ["get", "post", "put", "delete", "patch"]:
                witnesses.add(method)
                
            # Security
            if "security" in details or "security" in spec:
                # Global security or per-operation
                secs = details.get("security", spec.get("security", []))
                if secs:
                    witnesses.add("auth_required")
                    # Check types if defined in components
                    # Simple heuristic for now: just presence implies auth
                else:
                    witnesses.add("no_auth")
            else:
                 witnesses.add("no_auth")

            # Data / Body
            if "requestBody" in details:
                content = details["requestBody"].get("content", {})
                if "application/json" in content: witnesses.add("json_body")
                if "multipart/form-data" in content: witnesses.add("file_upload")
                
            # PII Detection (Heuristic on parameter names or schema properties)
            # Flatten everything to search words
            details_str = json.dumps(details).lower()
            if "email" in details_str: witnesses.add("pii_email")
            if "ssn" in details_str or "socialsecurity" in details_str: witnesses.add("pii_ssn")
            if "credit_card" in details_str or "cc_number" in details_str: witnesses.add("pii_credit_card")
            
            # Pagination check
            if "limit" in details_str and "offset" in details_str:
                witnesses.add("pagination")
                
            # Required Field Check
            # Look inside requestBody -> content -> application/json -> schema -> required
            if "requestBody" in details:
                content = details["requestBody"].get("content", {})
                json_content = content.get("application/json", {})
                schema = json_content.get("schema", {})
                if "required" in schema and schema["required"]:
                    witnesses.add("required_field")

    return witnesses
