# REUG Runtime Quick Start

## Quick start (local)

1. **Environment**

   Set one provider key in your environment or `.env` file: `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`.
   Optional knobs default safely:

   - `REUG_EVENTBUS` (`file` or `redis`)
     - `REUG_EVENT_LOG_DIR` (default `./logs/events`)
     - `REUG_TOOL_REGISTRY_DIR`
     - `REUG_MAX_TOOL_CALLS`, `REUG_EXEC_TIMEOUT_S`, `REUG_EXEC_MAX_RETRIES`
     - `API_PREFIX` (default `/`) – optional path prefix for all API routes
   - `MCP_BROADCAST_URL` (optional MCP telemetry endpoint)
   - `MCP_BROADCAST_TOKEN` (bearer token for MCP_BROADCAST_URL)

   If `REUG_EVENTBUS` is unset or a Redis backend is unavailable, the runtime
   gracefully falls back to appending JSONL telemetry under
   `./logs/events/events.jsonl`.

   If `.env` is missing:

   ```bash
   cp .env.example .env   # then edit only the API key you use
   ```

2. **Install & run**

   ```bash
   make deps               # pip install -r requirements.txt -r requirements-test.txt
   # add GPU support: pip install -r requirements-gpu.txt
   make lint               # run pre-commit hooks
   make test-smoke         # quick runtime smoke test
   make run                # uvicorn app:app --reload --port 8080
   ```

3. **Sanity check**

   ```bash
    curl -sf http://localhost:8080/healthz
      curl -sN -X POST http://localhost:8080${API_PREFIX}/v1/chat/stream \
      -H 'content-type: application/json' \
      -d '{"message":"hello world","session_id":"local"}'
   ```

## Quick start (Docker)

With `Dockerfile` and `docker-compose.yml` present:

```bash
docker compose --profile redis up --build -d redis   # optional Redis
docker compose up --build app
curl -sf http://localhost:8080/healthz
```

## Endpoints

- `POST ${API_PREFIX}/v1/chat/stream` – streamed agent turn
- `POST ${API_PREFIX}/tools/*` – toolbox endpoints
- `GET /healthz` – health probe

## Codex automation

Paste the following task in Codex to bootstrap, test, and verify the runtime:

```
Goal: Boot the runtime server defined in app.py with the REUG streaming router and toolbox.

Steps:
1) Environment
   - create .env from .env.example if missing; ensure PYTHONPATH=./src is set.
   - append GEMINI_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY if present.
   - install deps: pip install -r requirements.txt -r requirements-test.txt
     # add GPU support: pip install -r requirements-gpu.txt
   - run pre-commit on touched files.
2) Sanity checks
   - python -m pip show fastapi uvicorn
   - python -c "import reug_runtime, sys; print('reug_runtime OK')" or ensure src on PYTHONPATH
3) Launch the server
   - uvicorn app:app --port 8080 --host 0.0.0.0 &
   - wait for http://localhost:8080/healthz
4) Exercise endpoints
    - POST ${API_PREFIX}/v1/chat/stream with {"message":"hello","session_id":"c1"}
   - POST /tools/run-tests if available
5) Artifact
   - commit and open a PR with test output if code changed
```

This task will retry on failures and verify that streaming yields a `<final_answer>` block.
