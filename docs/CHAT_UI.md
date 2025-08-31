# Super Alita Chat UI

This repository ships with a minimal, modern chat interface backed by the streaming agent API.

Quick start:

- Install dependencies: `make deps` (or `uv pip install -r requirements.txt -c constraints.txt`)
- Run the FastAPI server: `make run` (serves `app:app` on port `8080`)
- Open the chat in your browser: `http://localhost:8080/`

Key endpoints:

- `GET /v1/chat/stream?q=...&session=...` — Server‑Sent Events (SSE) stream. Emits:
  - `start`: session + model identity + rate‑limit headers
  - `content`: incremental response tokens
  - `tool_start`: tool call metadata (name + args)
  - `tool_result`: tool execution result
  - `done`: completion marker
- `POST /v1/chat` — Non‑streaming JSON fallback
- `GET /v1/chat/history?session=...` — Retrieve session history
- `DELETE /v1/chat/history?session=...` — Clear session history

Notes:

- API key auth and rate limiting are configurable via environment variables (see `src/main.py`).
- The UI auto‑detects the API base from the browser origin and falls back to JSON if SSE is unavailable.
- The stream includes periodic heartbeat frames to keep proxies from closing idle connections.

