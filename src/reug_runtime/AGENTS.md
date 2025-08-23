# AGENTS.md (REUG Runtime Agent)

## Overview
This folder hosts the **deployable runtime agent**.  
Use this file for agent-specific commands and safety rules.

## Setup
```bash
uv pip install -r ../requirements.txt -c ../constraints.txt
python -m reug_runtime.main
```

## Dev / Run
- Health:
  ```bash
  curl http://127.0.0.1:8000/healthz
  ```
- Logs & traces go to `${SUPER_ALITA_DATA_DIR}/runs/<ts>/`

## Safety (Agent-Critical)
- No raw `eval`/`exec`: use `src/sandbox/exec_sandbox.py`
- YAML via `src/core/yaml_utils.py`
- Subprocess via `src/core/proc.py` (no `shell=True`)
- All credentials via env; never in repo

## Tests (Agent Scope)
```bash
pytest -q reug_runtime tests -k "runtime or sandbox"
```

## PR Gates
- `pre-commit run --all-files`
- `pytest`, coverage ≥ 70%

## Notes
- Prefer `shadow` mode for dry runs; use `act` for tool execution once tests pass.

