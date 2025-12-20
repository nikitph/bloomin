def generate_ts_api_endpoint(witnesses: set[str], with_proof=True) -> str:
    """
    Generates a full industry-standard TypeScript API endpoint from semantic witnesses.
    Targets: Express, Zod, and Typed Patterns.
    """
    lines = ["import { Router, Request, Response } from 'express';"]
    if "zod_schema" in witnesses:
        lines.insert(1, "import { z } from 'zod';")
    
    trace = []
    
    # 1. Validation Schema
    if "zod_schema" in witnesses:
        lines.append("\n// Validation Schema")
        schema_props = ["  name: z.string().min(3),"]
        if "optional_field" in witnesses: # Inheriting from base TS vocab if present
            schema_props.append("  email: z.string().email().optional(),")
        if "resource_id" in witnesses:
            schema_props.append("  id: z.string().uuid(),")
            
        lines.append(f"const UserSchema = z.object({{\n{chr(10).join(schema_props)}\n}});")
        trace.append("zod_schema -> generated Zod validation object")

    # 2. Interface
    if "typed_interface" in witnesses:
        lines.append("\ninterface UserRequest extends Request {")
        lines.append("  body: z.infer<typeof UserSchema>;")
        lines.append("}")
        trace.append("typed_interface -> generated Request extension")

    # 3. Router Setup
    lines.append("\nconst router = Router();")
    
    # 4. Middleware chain
    middleware = []
    
    # PARAMETERIZED WITNESS HANDLING
    for w in witnesses:
        if w.startswith("role:"):
            role = w.split(":")[1]
            middleware.append(f"checkRole(['{role}'])")
            trace.append(f"parameterized_role -> injected role check for {role}")
        if w.startswith("auth:"):
            mech = w.split(":")[1]
            middleware.append(f"{mech}AuthMiddleware")
            trace.append(f"parameterized_auth -> injected {mech} authentication")

    if not any(w.startswith("role:") for w in witnesses) and "role_protected" in witnesses:
        middleware.append("checkRole(['admin'])")
    if not any(w.startswith("auth:") for w in witnesses) and "jwt_auth" in witnesses:
        middleware.append("authMiddleware")
    
    mw_str = ", ".join(middleware) + ", " if middleware else ""

    
    # 5. Method & Route
    method = "post"
    if "http_get" in witnesses: method = "get"
    elif "http_put" in witnesses: method = "put"
    elif "http_delete" in witnesses: method = "delete"
    
    route = "/users"
    if "route_params" in witnesses: route += "/:id"
    
    handler_prefix = "async " if "async_handler" in witnesses else ""
    
    lines.append(f"\nrouter.{method}('{route}', {mw_str}{handler_prefix}(req: Request, res: Response) => {{")
    
    # 6. Handler Body
    body = []
    if "error_boundary" in witnesses:
        body.append("  try {")
        indent = "    "
    else:
        indent = "  "
        
    if "body_validation" in witnesses:
        body.append(f"{indent}const validated = UserSchema.parse(req.body);")
        trace.append("body_validation -> added Schema.parse() call")

    # HEURISTIC: Shifting implementation based on Performance/Cost witnesses
    if "low_latency" in witnesses:
        body.append(f"{indent}// [HEURISTIC: LOW_LATENCY] - Using in-memory cache check")
        body.append(f"{indent}if (cache.has(validated.id)) return res.json(cache.get(validated.id));")
        trace.append("low_latency -> injected caching layer")

    if "db_save" in witnesses:
        if "batch_processing" in witnesses:
            body.append(f"{indent}// [HEURISTIC: BATCH] - Using bulk persistence")
            body.append(f"{indent}await db.user.createMany({{ data: [validated] }});")
            trace.append("batch_processing -> used high-throughput bulk insert")
        else:
            body.append(f"{indent}await db.user.create({{ data: validated }});")
            trace.append("db_save -> added standard DB persistence")

    if "json_response" in witnesses:
        res_payload = "{ success: true }"
        if "pagination_metadata" in witnesses:
            res_payload = "{ success: true, meta: { total: 100, page: 1 } }"
        body.append(f"{indent}res.json({res_payload});")
        trace.append("json_response -> added API response handler")
        
    if "error_boundary" in witnesses:
        body.append("  } catch (err) {")
        body.append("    res.status(400).json({ error: err.message });")
        body.append("  }")
        trace.append("error_boundary -> wrapped in try-catch block")

    lines.extend(body)
    lines.append("});")
    
    output = "\n".join(lines)
    
    if with_proof:
        proof = "/**\n * [WITNESS_TRACE]\n" + "\n".join(f" * - {t}" for t in trace) + "\n */\n"
        output = proof + "\n" + output
        
    return output
