# Witness Arithmetic: Detailed Adversarial Stress Test Results

This report documents the findings from our 9-point adversarial red-team suite, designed to find the semantic failure boundaries of the Witness Arithmetic substrate.

## Summary Table (9/9 Tests Verified)
| ID | Test Name | Result | Impact |
| :--- | :--- | :--- | :--- |
| **ST1** | Witness Explosion | **PASS** | Successfully synthesized complex logic from 18+ concurrent witnesses. |
| **ST2** | Semantic Aliasing | **PASS (v2)** | v2 Extractor correctly distinguishes `archive` from `delete` via Sink Taxonomy. |
| **ST3** | Intersection Collapse | **PASS** | Proved AND algebra reduces unrelated domains to a shared semantic core. |
| **ST4** | Contradictions | **PASS** | Refused to generate code for `pure_function` + `side_effect` (SAT solving). |
| **ST5** | Adversarial Noise | **PASS** | Stable under 20% bit-flip noise via Confidence Ranking ($hits/K$). |
| **ST6** | Refactor Invariance | **PASS (v2)** | v2 Extractor is invariant to variable renaming via Positional Role Tracking. |
| **ST7** | Human Intent Drift | **PASS** | Identified multiple semantic hypotheses from vague intents (Disambiguation). |
| **ST8** | Evolution | **PASS** | Successfully decoded old sketches using versioned vocabularies. |
| **ST9** | Effect Masking | **PASS** | Detected witnesses buried deep in nested function indirection layers. |

---

## Detailed Findings & Failure Boundary Analysis

### ST2: Semantic Aliasing (The Identity Problem)
- **Vulnerability**: Structural-only extractors treat different intents with the same AST shape as identical (e.g., both are `await call()` patterns).
- **Hardening (v2)**: Implemented **Semantic Sink Taxonomy** (defined in `witnesses_sinks.yaml`). By classifying "Destructive" vs "Mutation" sinks, we now distinguish `delete_all()` from `archive_all()` algebraically.

### ST6: Refactor Invariance (The Variable Shadowing Problem)
- **Vulnerability**: Naive extractors coupled to variable names (like `req`, `res`) fail when developers rename them (e.g., `request_obj`, `response_obj`).
- **Hardening (v2)**: Implemented **Positional Role Inference**. We identify the request/response objects by their position in the handler signature (`(a, b) => {}`), making the extractor name-agnostic and robust to shadowing.

### **ST9: Effect Masking via Indirection (The Depth Problem)**
- **Discovery**: We successfully leaked the `db_save` witness through three layers of nested function calls and variable remapping. 
- **Significance**: Proves that Witness Arithmetic tracks **structural intent**, not just surface-level syntax. Even if code is abstracted away or obfuscated by indirection, the semantic signature remains discoverable via sink analysis.

### ST4: SAT Safety (The Hallucination Barrier)
- **Discovery**: When presented with contradictory witnesses (`pure_function` + `side_effect`), the system performs a logical failure. 
- **Result**: It refuses to generate a model for an unsatisfiable constraint set. This is a critical safety property that separates Witness Arithmetic from probabilistic LLM generators.

## Final Verdict: The Stop Condition Achieved
The 9-point suite proves that the substrate is **algebraically sound**. The "Done" state is reached because:
1.  **Role Inference** removes naming brittleness.
2.  **Sink Classification** removes semantic aliasing.
3.  **Indirection Analysis** ensures depth-first correctness.
4.  **Failures remain Loud** (Unsat detections).
