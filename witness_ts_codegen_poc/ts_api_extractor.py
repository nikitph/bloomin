import re

def extract_witnesses_ts_api(code: str) -> set[str]:
    """
    Extracts semantic witnesses from TypeScript API code (Express + Zod).
    Focuses on structural patterns and library-specific usage.
    """
    witnesses = set()
    
    # 1. Transport (HTTP Methods)
    if re.search(r"\.get\(", code): witnesses.add("http_get")
    if re.search(r"\.post\(", code): witnesses.add("http_post")
    if re.search(r"\.put\(", code): witnesses.add("http_put")
    if re.search(r"\.delete\(", code): witnesses.add("http_delete")
    if ":" in code and ("router." in code or "app." in code): witnesses.add("route_params")
    if "req.query" in code: witnesses.add("query_params")

    # 2. Validation (Zod)
    if "z.object" in code: witnesses.add("zod_schema")
    if "req.body" in code and ("parse" in code or "validate" in code): witnesses.add("body_validation")
    if "req.params" in code and ("parse" in code or "validate" in code): witnesses.add("param_validation")
    
    # 3. Security
    if "auth" in code.lower() or "jwt" in code.lower(): witnesses.add("jwt_auth")
    if "role" in code.lower() or "permission" in code.lower(): witnesses.add("role_protected")
    if "x-api-key" in code.lower(): witnesses.add("api_key_required")
    if "cors" in code.lower(): witnesses.add("cors_enabled")

    # 4. Logic
    if "async" in code: witnesses.add("async_handler")
    if "class " in code and "Controller" in code: witnesses.add("controller_method")
    if "try {" in code and "catch" in code: witnesses.add("error_boundary")
    if "return " in code and "if (" in code: witnesses.add("early_return")
    if ".save(" in code or "db." in code or "prisma" in code: witnesses.add("db_save")

    # 5. Data
    if "res.json" in code or "res.send" in code: witnesses.add("json_response")
    if "interface " in code or "type " in code: witnesses.add("typed_interface")
    if "limit" in code or "offset" in code or "page" in code: witnesses.add("pagination_metadata")
    if "id" in code.lower() and "uuid" in code.lower() or "mongo" in code.lower(): witnesses.add("resource_id")

    return witnesses
