import re

def extract_witnesses_ts(code: str) -> set[str]:
    """
    Extracts semantic witnesses from TypeScript code using structural heuristics.
    (Simulating a TypeScript AST parser for the POC)
    """
    witnesses = set()
    
    # Types
    if "interface " in code: witnesses.add("typed_interface")
    if re.search(r"<[A-Z][a-zA-Z0-9]*>", code): witnesses.add("generic_type")
    if " | " in code: witnesses.add("union_type")
    if "?:" in code: witnesses.add("optional_field")
    
    # Control Flow
    if "async " in code or "await " in code: witnesses.add("async_await")
    if "for " in code or "while " in code: witnesses.add("loop")
    if "if " in code or "else " in code: witnesses.add("conditional")
    if "return" in code and "if" in code: witnesses.add("early_return")
    
    # Safety / Logic
    if "throw " in code or "try {" in code: witnesses.add("error_handling")
    if "fetch(" in code or "axios" in code: witnesses.add("api_call")
    if ".save(" in code or "db." in code: witnesses.add("db_save")
    if "limit" in code and "offset" in code: witnesses.add("pagination")
    if "validator" in code or "validate" in code: witnesses.add("input_validation")

    return witnesses
