# Witness Arithmetic: Production Roadmap (Path to 1.0)

This document outlines the remaining engineering requirements to transition the current POC into a production-grade system.

## 1. Witness Taxonomy Expansion
*   **Hierarchical Witnesses**: Support parent-child relationships (e.g., `db_persistence` -> `postgres_jsonb`).
*   **Parameterized Witnesses**: Moving from boolean tokens to parameterized signatures: `pagination(size: number)`, `auth(roles: string[])`.
*   **Conflict Resolution**: Rules for handling mutually exclusive witnesses (e.g., `serverless` vs `long_running_process`).

## 2. Synthesis Hardening
*   **AST Manipulation**: Move from string-based templates to a first-class AST library like `ts-morph` or the `TypeScript Compiler API`.
*   **Heuristic Overlays**: Implement sophisticated "Complexity Shifting" where Performance/Cost witnesses rewrite the entire synthesis strategy.
*   **Formatters**: Integrated Prettier/ESLint rules to ensure syntactic perfection.

## 3. Production Extractors
*   **Zero-False-Positives**: Replace heuristic matching with strict AST-visiting.
*   **Language Support Expansion**:
    *   **Terraform/HCL**: For Infrastructure-as-Code.
    *   **Solidity/Rust**: For Smart Contracts.
    *   **Protobuf/gRPC**: For cross-service contracts.

## 4. Performance & Scale
*   **Sketch Optimization**: Switch from standard Bloom Filters to **Invertible Bloom Lookup Tables (IBLTs)** for exact, collision-free inversion at scale.
*   **Vector Search Integration**: Store sketches in HNSW or FAISS for million-scale semantic searching.
*   **O(1) Delta Streams**: Differential updates for sketches in high-throughput environments.

## 5. Ecosystem & Governance
*   **CI/CD Guardrails**: "Semantic Regressions" (e.g., fail build if `encryption_at_rest` is removed).
*   **Observability**: Witness-based tracing and logs.
*   **AI Integration**: Using LLMs strictly as "Intent to Witness" translators, keeping the final generation deterministic.

---

### The Vision
A world where code is not "written" but **solved** from a set of semantic constraints.
