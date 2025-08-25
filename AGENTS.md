# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/` (planner, sandbox, plugins, telemetry, orchestration). Entry: `src/main.py`.
- API dev app: `app.py` (served by `uvicorn` via `make run`).
- Tests: `tests/` (plus some top-level `test_*.py`). Mirror `src/` layout.
- Config/docs/tools: `config/`, `docs/`, `extensions/`, `tools/`, `docker/`.

## Build, Test, and Development Commands
- Install deps: `uv pip install -r requirements.txt -c constraints.txt` (or `make deps`).
- Run runtime server: `python -m src.main`.
- FastAPI dev server: `make run` (serves `app:app` on port 8080).
- Tests: `pytest -q` (filter: `pytest -k "expr"`; marker: `-m integration_redis`).
- Hooks/format: `pre-commit run --all-files`.

## Coding Style & Naming Conventions
- Python 3.11+. Format with `black` (88 cols) and lint with `ruff`.
- Type-check with `mypy --strict` (especially `src/core`, `src/sandbox`).
- Use double quotes; add type hints; keep functions small and pure.
- No raw `eval/exec`; use `src/sandbox/exec_sandbox.py`.
- Subprocess/YAML: `src/core/proc.py` (no `shell=True`) and `src/core/yaml_utils.py`.

## Testing Guidelines
- Framework: `pytest`; target ≥70% coverage. New code requires tests.
- Naming: files `test_*.py`; organize to mirror `src/` packages.
- Useful: `pytest -q`, `pytest -k name`, `pytest -m integration_redis`.

## Commit & Pull Request Guidelines
- Commits: `[module] Short description` (e.g., `[sandbox] Harden exec policy`).
- Before PR: run hooks, type-check, and tests; CI enforces lint/type/test/coverage.
- PRs: include summary, rationale, linked issues, and updated docs/config.
- Secrets: never commit keys; manage via env or `.env` (see `.env.example`).

## Security & Run Modes
- All dynamic execution must be sandboxed; do not bypass policy guards.
- Configure via `SUPER_ALITA_MODE`: `shadow` (plan), `act` (sandboxed act), `batch` (replay).
