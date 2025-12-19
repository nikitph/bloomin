import re
import yaml
from pathlib import Path

def load_sinks():
    sink_path = Path(__file__).parent / "witnesses_sinks.yaml"
    with open(sink_path, "r") as f:
        return yaml.safe_load(f)

SINKS = load_sinks()

def extract_witnesses_v2(code: str) -> set[str]:
    """
    Extractor v2: Uses Positional Role Tracking and Sink Analysis.
    Solves ST2 (Aliasing) and ST6 (Name Coupling).
    """
    witnesses = set()
    
    # 1. POSITIONAL ROLE TRACKING
    # Support both: (req, res) => { ... } AND function(req, res) { ... }
    handler_patterns = [
        r"function\s*\w*\s*\(([\w\d_]+),\s*([\w\d_]+)\)", # function(a, b)
        r"\(([\w\d_]+),\s*([\w\d_]+)\)\s*=>"              # (a, b) =>
    ]
    
    req_name, res_name = None, None
    for pattern in handler_patterns:
        match = re.search(pattern, code)
        if match:
            req_name = match.group(1)
            res_name = match.group(2)
            break
            
    if req_name:
        # Now track 'req_name' usage instead of hardcoded 'req'
        if f"{req_name}.body" in code: witnesses.add("body_validation")
        if f"{req_name}.params" in code: witnesses.add("param_validation")
        if f"{req_name}.query" in code: witnesses.add("query_params")
        
    if res_name:
        # Track 'res_name' usage instead of hardcoded 'res'
        if f"{res_name}.json" in code or f"{res_name}.send" in code: witnesses.add("json_response")

    # 2. SINK ANALYSIS (Solving ST2)
    # Check for destructive vs non-destructive sinks using word boundaries or sub-calls
    for sink in SINKS['sinks']['destructive']:
        if sink in code and f"{sink}(" in code: 
            witnesses.add("sink_destructive")
            break
            
    for sink in SINKS['sinks']['mutation']:
        if sink in code and f"{sink}(" in code:
            witnesses.add("db_save")
            break
            
    for sink in SINKS['sinks']['external']:
        if sink in code and f"{sink}(" in code:
            witnesses.add("sink_external")
            break


    # 3. BASE TRANSPORT & LOGIC (Robust Patterns)
    if re.search(r"\.get\(", code): witnesses.add("http_get")
    if re.search(r"\.post\(", code): witnesses.add("http_post")
    if re.search(r"\.put\(", code): witnesses.add("http_put")
    if re.search(r"\.delete\(", code): witnesses.add("http_delete")
    if ":" in code and (".post" in code or ".get" in code): witnesses.add("route_params")
    
    if "async" in code: witnesses.add("async_handler")
    if "try {" in code and "catch" in code: witnesses.add("error_boundary")
    if "auth" in code.lower() or "jwt" in code.lower(): witnesses.add("jwt_auth")
    if "z.object" in code: witnesses.add("zod_schema")

    return witnesses
