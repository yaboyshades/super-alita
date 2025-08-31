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
