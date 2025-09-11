# Constitutional Coding Architect v3.5-C

**This workspace enforces a strict, test-first coding workflow.**

**Your Rules (Copilot):**
1.  **Source of Truth:** The CSV files (`10_requirements_ledger.csv`, `11_dependencies_manifest.csv`) are the source of truth. Do not invent requirements.
2.  **Workflow:** Follow the sequence: **Requirement -> Test -> Implementation -> Gate**.
3.  **Test-First:** Always generate failing tests in `tests/` *before* writing implementation code in `src/`.
4.  **Gate Enforcement:** Do not consider a feature complete until it passes the **Coding Gate** with a score of ≥85, as defined in `01_CODING_GATE_RUBRIC.md`. Log all gate attempts in `99_gate_log.csv`.
5.  **Schema Compliance:** All CSV outputs must comply with the structure defined in `90_schema.json`.

