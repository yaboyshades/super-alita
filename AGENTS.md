# AGENTS.md

## Project Overview
Super Alita is a cognitive agent runtime that wires together:
- A **planner** (LADDER/AOG reasoning, neural atoms)
- **Plugins** (tools and skills)
- A **sandbox** for safe dynamic code execution
- **Telemetry** (event bus, Prometheus)
- Integrations with **MCP server** and **VS Code Agent Mode**

This file gives coding agents *mechanical instructions*: build, test, style, and safety guardrails. See `README.md` for human quickstart.

## Setup Commands
- Install deps:
  ```bash
  uv pip install -r requirements.txt -c constraints.txt
  ```
- Start runtime server:
  ```bash
  python -m src.main
  ```
- Run tests:
  ```bash
  pytest -q --maxfail=1 --disable-warnings
  ```

## Code Style
- **Python 3.11+**
- `ruff` + `black` (`88` cols). Quotes: double
- `mypy --strict` on `src/core` and `src/sandbox`
- No raw `eval`/`exec` → use `src/sandbox/exec_sandbox.py`
- YAML via `src/core/yaml_utils.py`
- Subprocess via `src/core/proc.py` (no `shell=True`)

## Testing Instructions
- Run all:
  ```bash
  pytest
  ```
- Coverage floor: 70%
- Slow tests marked `-m slow`
- Focus a test:
  ```bash
  pytest -k "name"
  ```
- New code requires tests
- Sandbox requires security tests under `tests/sandbox/`

## PR / Commit
- Title: `[module] Short description`
- Run `pre-commit run --all-files` before commit
- Secrets are blocked by pre-commit; never commit keys
- CI enforces lint/type/test/coverage

## Run Modes
- `shadow`: Plan only
- `act`: Plan + act under sandbox
- `batch`: Replay traces

Set via `SUPER_ALITA_MODE`.

## Safety
- All dynamic execution sandboxed
- Secrets from env/.env only
- Subprocess + YAML sanitized by core helpers

## Large Monorepo Note
Subprojects may include their own `AGENTS.md`. The closest file to the edited path wins.

