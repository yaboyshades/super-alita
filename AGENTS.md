# Repository Guidelines

## Project Structure & Module Organization

- Source code: `src/` (planner, sandbox, plugins, telemetry, orchestration). Entry point: `src/main.py`.
- API dev app: `app.py` (served by `uvicorn` via `make run`).
- Tests: `tests/` plus top-level `test_*.py`; mirror `src/` layout.
- Config/docs/tools: `config/`, `docs/`, `extensions/`, `tools/`, `docker/`.

## Build, Test, and Development Commands

- Install deps: `uv pip install -r requirements.txt -c constraints.txt` (or `make deps`).
- Run runtime server: `python -m src.main`.
- FastAPI dev server: `make run` (serves `app:app` on port 8080).
- Tests:
  - `pytest -q`
  - `pytest -q -k "expr"`
  - `pytest -q -m integration_redis`
- Lint/format: `ruff check .` and `black . -l 88` (or `pre-commit run --all-files`).
- Type-check: `mypy --strict src core` (focus on `src/core`, `src/sandbox`; add `app.py` as needed).

## Coding Style & Naming Conventions

- Python 3.11+. Use 4-space indentation, double quotes, and explicit type hints.
- Keep functions small and pure; avoid side effects.
- No raw `eval/exec`; use `src/sandbox/exec_sandbox.py` for dynamic code.
- Subprocess/YAML: use `src/core/proc.py` (no `shell=True`) and `src/core/yaml_utils.py`.
- Naming: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.

## Testing Guidelines

- Framework: `pytest`; target ≥70% coverage for changes.
- Naming: files `test_*.py`; structure tests to mirror `src/` packages.
- Useful patterns: `pytest -k name`, `pytest -m integration_redis`.
- Write unit tests for new modules and critical paths; prefer fast, isolated tests.

## Commit & Pull Request Guidelines

- Commits: `[module] Short description` (e.g., `[sandbox] Harden exec policy`).
- Before PR: run hooks, type-check, and tests; CI enforces lint/type/test/coverage.
- PRs: include summary, rationale, linked issues, and updated docs/config when applicable.
- Secrets: never commit keys; manage via env or `.env` (see `.env.example`).

## Security & Run Modes

- All dynamic execution must be sandboxed; do not bypass policy guards.
- Process/YAML must go through repository utilities (`proc.py`, `yaml_utils.py`).
- Modes via `SUPER_ALITA_MODE`: `shadow` (plan), `act` (sandboxed act), `batch` (replay).

## Spec‑Kit SDD Workflow (Integrated)

Spec‑Driven Development (Spec‑Kit) is a first‑class workflow in this repo. It provides a consistent path from specification → plan → tasks, with constitutional validation and test‑first gates.

### What’s included

- FastAPI endpoints:
  - `POST /sdd/specify`
  - `POST /sdd/plan`
  - `POST /sdd/tasks`
- Key runtime files:
  - `src/sdd/router.py` — FastAPI routes for SDD
  - `src/sdd/models.py` — Pydantic request/response models
  - `src/sdd/enhanced_sdd_framework.py` — SDD pipeline logic (with Mangle integration)
  - `src/sdd/config.py` — SDD configuration and defaults
  - `src/sdd/validators.py` — Constitutional compliance checks
  - `src/orchestration/unified_orchestrator.py` — Orchestrator wired for SDD + reliability
- Templates & memory:
  - `templates/sdd/spec-template.md`
  - `templates/sdd/plan-template.md`
  - `templates/sdd/tasks-template.md`
  - `memory/sdd/constitutional_sdd_framework.md`

### How to run (Windows PowerShell)

1. Start the API (development):

```powershell
uvicorn app:app --reload --port 8080
```

1. Call SDD endpoints:

```powershell
# /sdd/specify
curl -X POST "http://127.0.0.1:8080/sdd/specify" `
  -H "Content-Type: application/json" `
  -d '{
    "user_input": "Add an SDD pipeline with constitutional validation gates.",
    "context": {"priority": "high"}
  }'

# /sdd/plan
curl -X POST "http://127.0.0.1:8080/sdd/plan" `
  -H "Content-Type: application/json" `
  -d '{
    "feature_id": "feat-sdd-pipeline"
  }'

# /sdd/tasks
curl -X POST "http://127.0.0.1:8080/sdd/tasks" `
  -H "Content-Type: application/json" `
  -d '{
    "feature_id": "feat-sdd-pipeline"
  }'
```

1. Use the CLI (sync wrappers around async SDD calls):

```powershell
# Specify → Plan → Tasks
python -m src.sdd.sdd_cli specify "Implement streaming SDD endpoints" --context '{"owner":"platform"}'
python -m src.sdd.sdd_cli plan feat-sdd-pipeline
python -m src.sdd.sdd_cli tasks feat-sdd-pipeline
```

### PowerShell Profile Setup for Spec Kit Integration

To enable seamless Spec Kit workflows in PowerShell, add the following to your PowerShell profile (`$PROFILE`):

```powershell
function Invoke-SpecKitWorkflow {
    param([string]$FeatureName, [string]$Phase = "all")
    switch ($Phase) {
        "constitution" { uvx spec-kit constitution $FeatureName }
        "specify" { uvx spec-kit specify $FeatureName }
        "plan" { uvx spec-kit plan $FeatureName }
        "tasks" { uvx spec-kit tasks $FeatureName }
        "implement" { uvx spec-kit implement $FeatureName }
        "all" {
            uvx spec-kit constitution $FeatureName
            uvx spec-kit specify $FeatureName
            uvx spec-kit plan $FeatureName
            uvx spec-kit tasks $FeatureName
            uvx spec-kit implement $FeatureName
        }
    }
}
Set-Alias spec Invoke-SpecKitWorkflow
```

This allows you to run SDD workflows with simple commands like:

- `spec "Add user authentication" plan` - Generate plan for user authentication feature
- `spec "Improve API performance" all` - Run full SDD pipeline for API performance improvement

### VS Code tasks (quick checks)

- SDD: Validate Environment — ensures key env vars are present
- SDD: Check Runtime — simple health check against the running server
- Run Prompt Pipeline — executes the prompt pipeline for ad‑hoc testing

Use from Command Palette: “Tasks: Run Task”.

### Quality gates and policies

- Constitutional threshold: overall compliance score ≥ 0.75
- Test‑first convention: unit tests for new modules and critical paths
- Simplicity Gate: small, focused functions; avoid unnecessary complexity
- Integration‑first verification for orchestrated flows
- Security: dynamic execution via `src/sandbox/exec_sandbox.py`; subprocess via `src/core/proc.py` (no `shell=True`); YAML via `src/core/yaml_utils.py` (safe loading)

## Unified Intelligence Layer Integration

The Unified Intelligence Layer is integrated into the Super-Alita ecosystem:

- **Contracts**: `src/unified_intelligence/contracts.yaml` - Defines request/response schemas
- **Orchestrator**: `src/unified_intelligence/orchestrator.py` - Implements fusion logic and canonical orchestration
- **Golden Fixtures**: `src/unified_intelligence/golden_fixtures.py` - Contains test fixtures for validation
- **Telemetry**: `src/unified_intelligence/telemetry.py` - Handles telemetry collection and envelopes
- **Validation Checklist**: `src/unified_intelligence/validation_checklist.py` - Comprehensive quality assurance

## Mangle Reasoning Engine Integration

The Mangle Reasoning Engine provides code analysis capabilities:

- **Code Ingester**: `src/unified_intelligence/code_reasoning/ingester.py` - AST-based symbol extraction
- **Rule Engine**: `src/unified_intelligence/code_reasoning/rules.py` - Datalog-like rule application
- **Models**: `src/unified_intelligence/code_reasoning/models.py` - Data structures for analysis
- **Facts Database**: SQLite storage for code relationships and dependencies

## Quality Assurance Integration

- **Constitutional Gates**: ≥75% compliance required for all changes
- **Code Quality Gates**: Complexity <10, 0 circular dependencies, ≥70% test coverage
- **Performance Gates**: Algorithmic complexity documented and validated
- **Integration Gates**: Compatibility with Super-Alita ecosystem verified

## Development Workflow Integration

- **SDD Pipeline**: Integrated with unified intelligence for specification → plan → tasks
- **Test-First**: Unit tests for new modules and critical paths
- **Contract-First**: Interface definitions before implementation
- **Fusion Logic**: Score combination for intelligent decision routing

### Notes

- The unified intelligence layer uses the reliability manager for retries/backoff
- Mangle reasoning integrates with the sandbox for secure code analysis
- All dynamic execution must be sandboxed; never bypass policy guards
- Telemetry flows through the unified collector for observability
- The SDD pipeline is integrated into the unified orchestrator and uses the reliability manager (retries, backoff, classification) under the hood.
- If repo‑wide linting is noisy due to tools/examples, scope checks to `src/` and core tests first.
