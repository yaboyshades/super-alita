# Runtime Environment Check

This note records a quick verification of the runtime server.

## Environment file
- `.env` copied from `.env.example` and contains runtime guardrail limits:
  - `REUG_MAX_TOOL_CALLS`
  - `REUG_EXEC_TIMEOUT_S`
  - `REUG_MODEL_STREAM_TIMEOUT_S`
  - `REUG_EXEC_MAX_RETRIES`
  - `REUG_RETRY_BASE_MS`
  - `REUG_SCHEMA_ENFORCE`
- Ensure API keys remain commented or unset unless needed; do **not** commit secrets.

## Dependency install
- `make deps` succeeded with no missing packages.
- `pre-commit` was not installed by default; install with `pip install pre-commit` to run project hooks.
- If dependencies fail to install, rerun `make deps` after adjusting the local Python environment.

## Server startup
- `python -m src.main` starts the FastAPI runtime and logs `runtime startup`.
- Use `Ctrl+C` to stop the server when testing locally.

No additional configuration or packages were required in this check.
