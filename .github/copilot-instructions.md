---
# Copilot Agent Instructions — Super Alita Project

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

## Key Patterns & Integration Points
- **SDD endpoints:** `POST /sdd/specify`, `/sdd/plan`, `/sdd/tasks` (see `src/sdd/router.py`)
- **Streaming orchestration:** `src/reug_runtime/router.py` (`POST /v1/chat/stream`)
- **Tool registry:** `src/tools/`, dynamic registry, tool execution via `/ability/execute/{tool_id}`
- **VS Code extension:** Custom commands for SDD, orchestrator, and consensus (see `src/vscode_integration/`)
- **Security:** All credentials via env; never in repo. Resource limits enforced in sandbox.

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
**For all changes: document constitutional compliance, prefer existing solutions, and keep functions small, pure, and well-typed.**
