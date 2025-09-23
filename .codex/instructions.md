---
# Copilot Agent Instructions — Super Alita Project

## CONSTITUTIONAL THINKING FRAMEWORK (ALWAYS ACTIVE)
**Every interaction must follow this cognitive process:**

### 1. Constitutional Article Assessment (Automatic)
- **Library-First**: Search existing solutions before proposing new code
- **Test-First**: Generate tests before/alongside implementation
- **Simplicity**: Functions ≤50 lines, complexity ≤10, single responsibility
- **Integration**: Validate within Super-Alita ecosystem context
- **Clarity**: Eliminate ambiguity, provide clear examples and documentation
- **Counterfactual**: Justify chosen approach over alternatives

### 2. Quality Gate Pre-Check (Every Code Output)
- **Mutation Resilience**: Will ≥90% of logical mutations be caught by tests?
- **CFG Uniqueness**: Does this create duplicate control flow patterns?
- **Formal Contracts**: Are pre/post conditions and invariants defined?
- **Property Coverage**: Are mathematical properties tested symbolically?
- **Performance**: Is algorithmic complexity documented and optimal?
- **Coverage**: Will this achieve ≥70% test coverage?
- **Calculus Gate**: Are runtime derivatives within mathematical bounds?

### 3. SDD Compliance (For Features/Changes)
- **Specification**: Clear, unambiguous requirements with research phase
- **Planning**: Architecture with complexity analysis and tradeoffs
- **Tasks**: Implementation breakdown with constitutional validation
- **Threshold**: ≥75% constitutional compliance score required

## Constitutional & SDD-First Development
- **All code and docs must comply with `.github/CONSTITUTION.md` (6 articles: Library-First, Test-First, Simplicity, Integration, Clarity, Counterfactual Justification).**
- **Spec-Driven Development (SDD) is mandatory for new features:**
  - Use `/specify` → `/plan` → `/tasks` (CLI: `src/sdd/sdd_cli.py`, API: `/sdd/*` endpoints).
  - All SDD outputs must pass constitutional gates (≥0.75 compliance).
  - Templates: `templates/sdd/`, validation: `src/sdd/validators.py`.

## Architecture & Security
- Modular structure: `src/` (core, sandbox, orchestration, tools, utils, abilities, sdd, reug_runtime).
- **Sandbox all dynamic code:** Use `src/sandbox/exec_sandbox.py` (never raw `eval`/`exec`).
- Subprocesses: `src/core/proc.py` (no `shell=True`). YAML: `src/core/yaml_utils.py`.
- No mocks, placeholders, or `NotImplementedError`—all code must be production-grade.
- Use absolute imports from `src.*`.

## Build, Test, and Validation
- **Quickstart:**
  1. `python -m venv .venv && . .venv/Scripts/Activate.ps1`
  2. `pip install -r requirements.txt -r requirements-test.txt`
  3. `python validate_deployment.py` (primary validation; use over `pytest` for full suite)
  4. `uvicorn app:app --reload --port 8080`
- **Lint/type/test:** `ruff check .`, `mypy --strict src`, `pytest -q`, `pre-commit run --all-files`
- **VS Code tasks:** SDD: Validate Environment, SDD: Check Runtime, pytest, check-all

## Calculus Runtime Derivative Gate (Article XXI)
- **Purpose:** Mathematical analysis of function performance to detect regressions via derivative spikes
- **Usage:** `python .vscode/copilot-middleware/calculus_gate_cli.py <file> --function <name>`
- **Quality Gates:**
  - **Slope Gate**: |df/dn| ≤ slope_limit (default: 2.0) - prevents constant-time violations
  - **Curvature Gate**: |d²f/dn²| ≤ curvature_limit (default: 1.0) - detects complexity changes
  - **Lipschitz Gate**: sensitivity ≤ lipschitz_limit (default: 10.0) - ensures stability
- **Grading:** A (all pass), B (minor fails), F (major fails)
- **Integration:** CLI for development, CI for builds, MCP for real-time analysis
- **Artifacts:** JSON certificates saved to `artifacts/calculus_gate/` with full analysis

## Key Patterns & Integration Points
- **SDD endpoints:** `POST /sdd/specify`, `/sdd/plan`, `/sdd/tasks` (see `src/sdd/router.py`)
- **Streaming orchestration:** `src/reug_runtime/router.py` (`POST /v1/chat/stream`)
- **Tool registry:** `src/tools/`, dynamic registry, tool execution via `/ability/execute/{tool_id}`
- **VS Code extension:** Custom commands for SDD, orchestrator, and consensus (see `src/vscode_integration/`)
- **Security:** All credentials via env; never in repo. Resource limits enforced in sandbox.

## Reliability & LLM Clients
- LLM client configuration is centralized in `src/core/settings.py` with `compute_retry_schedule` producing deterministic backoff; honour `LLM_RETRY_MULTIPLIER` and `LLM_RETRY_JITTER_RATIO` (documented in `.env.example`).
- `src/reug_runtime/llm_client.py` and related adapters must consume `LLM_RETRY_SCHEDULE` for consistency; never inline retry loops.
- When adding providers, extend `src/core/settings.py` with typed constants and update tests in `tests/test_core_settings.py`.

## Observability & Event Contracts
- Emit canonical events defined in `src/orchestration/event_schemas.py` and validated against `docs/specs/unified_orchestration_p0_event_schema_spec.md`.
- Use `src/orchestration/observability.py` helpers for structured logs/metrics; avoid ad-hoc `print`.
- Persist run ledgers via the unified orchestrator when reliability scenarios require replay.

## Reliability Manager & Tooling
- Prefer `src/orchestration/reliability_manager.py` for retries/circuit breakers; legacy `src/core/reliability.py` is in sunset mode.
- Telemetry must flow through `src/core/telemetry/collector.py`; register new probes there instead of bespoke logging.
- Security gates live in `src/contracts/gates/`; augment them instead of bypassing constitutional checks.

## Elite Code Quality Patterns (Auto-Enforced)
- **Mutation Score Gate**: `.vscode/copilot-middleware/mutant_gate.py` flips comparisons/booleans and requires `pytest` to kill >=90% of generated mutants before code lands.
- **CFG Hash Deduplication**: `.vscode/copilot-middleware/cfg_hash_guard.py` hashes function control-flow graphs and blocks duplicates across the repo.
- **SyGuS Minimiser**: `.vscode/copilot-middleware/sygus_minimizer.py` simplifies functions tagged with `# sygus:minimize` using SymPy-backed synthesis.
- **Symbolic Execution**: Property-based tests that explore all code paths
- **Formal Verification**: Pre/post conditions and invariants with typed annotations
- **Performance Analysis**: Algorithmic complexity documentation and optimization
- **Invariant Detection**: Automated enforcement of null, bounds, and state constraints

## Agent Mode Cognitive Framework (Internal Process)
**This thinking process is automatic and always active:**

### Pre-Response Constitutional Filter
1. **Library Research**: Check `src/tools/`, `src/core/`, existing patterns before new code
2. **Test Strategy**: Plan pytest cases that will catch logical mutations (not just syntax)
3. **Complexity Analysis**: Ensure functions ≤50 lines, single responsibility, clear interfaces
4. **Integration Validation**: Verify compatibility with Super-Alita ecosystem
5. **Documentation Requirements**: Pre/post conditions, examples, clear parameter descriptions
6. **Alternative Analysis**: Consider and document why chosen approach over alternatives

### Quality Gate Pre-Validation
- **Mutation Resilience Check**: Would changing `==` to `!=`, `True` to `False`, `+` to `-` break tests?
- **CFG Uniqueness Scan**: Does this control flow already exist in the codebase?
- **Property Coverage**: Are mathematical invariants (monotonicity, commutativity, bounds) tested?
- **Performance Impact**: Is algorithmic complexity O(?) documented and justified?
- **Constitutional Compliance**: Does this achieve ≥75% compliance across all 6 articles?

### SDD Workflow Integration
- **For any substantial change**: Follow `/specify` → `/plan` → `/tasks` pipeline
- **Constitutional gates**: Each stage must pass ≥75% compliance threshold
- **Templates**: Use `templates/sdd/` for structured specifications
- **Validation**: All outputs validated by `src/sdd/validators.py`

## Examples & Conventions
- **Tool implementation:** See `src/tools/AGENTS.md` for base classes and patterns.
- **Event building:** Use `src/utils/event_builders.py`.
- **Testing:** Target ≥70% coverage; mirror `src/` in `tests/`.
- **No direct file access outside workspace; validate with guardrails (`src/utils/guardrails.py`).**

## Troubleshooting & Performance
- Use `python validate_deployment.py` for system health (preferred over full `pytest`).
- Linting: scope to `src/` if noisy. Makefile may have formatting issues—prefer direct shell commands.
- Known: Some tests may have import/syntax errors; focus on core validation.

---
**CORE PRINCIPLE: Every response must embody constitutional thinking and quality gate validation. This is not optional guidance—it's the fundamental cognitive framework that governs all interactions. Code without constitutional compliance and quality gates is considered incomplete and must be revised.**

**For all changes: document constitutional compliance, prefer existing solutions, and keep functions small, pure, and well-typed.**
