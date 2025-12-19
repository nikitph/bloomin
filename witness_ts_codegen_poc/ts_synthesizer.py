def synthesize_ts(witnesses: set[str], with_proof=False) -> str:
    """
    Synthesizes TypeScript code from a set of witnesses.
    Deterministic, structural synthesis.
    """
    lines = []
    trace = []

    # 1. Interfaces & Types (Foundational)
    if "typed_interface" in witnesses:
        name = "User"
        props = ["id: string;", "name: string;"]
        if "optional_field" in witnesses:
            props.append("email?: string;")
        
        lines.append(f"interface {name} {{\n  {chr(10).join('  ' + p for p in props)}\n}}")
        trace.append("typed_interface -> defined Interface 'User'")

    # 2. Implementation logic
    impl_lines = []
    
    # Safety first (Constraint-driven)
    if "input_validation" in witnesses:
        impl_lines.append("  if (!user.id) throw new Error('Invalid ID');")
        trace.append("input_validation -> added guard clause")
        
    if "authorization_check" in witnesses:
        impl_lines.append("  await checkAuth(user);")
        trace.append("authorization_check -> added auth hook")

    # Action logic
    if "db_save" in witnesses:
        if "generic_type" in witnesses:
            impl_lines.append("  await db.save<User>(user);")
        else:
            impl_lines.append("  await db.save(user);")
        trace.append("db_save -> added database persistence")
        
    if "side_effect" in witnesses:
        impl_lines.append("  console.log('User saved', user.id);")
        trace.append("side_effect -> added logging hook")

    # Wrap in function
    func_name = "saveUser"
    is_async = "async_await" in witnesses
    prefix = "async " if is_async else ""
    
    body = "\n".join(impl_lines) if impl_lines else "  // empty implementation"
    lines.append(f"{prefix}function {func_name}(user: User) {{\n{body}\n}}")
    if is_async: trace.append("async_await -> enabled async function modifier")

    output = "\n\n".join(lines)
    
    if with_proof:
        proof = "/**\n * [WITNESS_TRACE]\n" + "\n".join(f" * - {t}" for t in trace) + "\n */\n"
        output = proof + output
        
    return output
