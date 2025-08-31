# Super‑Alita User Guide

## Quick Start

- Start the dev server: `make run` (serves `app:app` on port 8080)
- Ping health: `curl http://127.0.0.1:8080/healthz`

## Query the API

- Non‑streaming:
  - `curl -X POST http://127.0.0.1:8080/api/v1/query -H "Content-Type: application/json" -d '{"prompt":"Hello"}'`
- Streaming (SSE):
  - `curl -N -X POST http://127.0.0.1:8080/api/v1/query -H "Content-Type: application/json" -d '{"prompt":"Hello","stream":true}'`

Fields:
- `prompt` (str): your prompt
- `mode` (str, optional): `hybrid` (default), `neural`, or `symbolic` (advisory)
- `session` (str, optional): session id for context
- `stream` (bool, optional): stream with SSE when true
- `max_tokens` (int, optional): advisory cap

Response (non‑streaming):
- `{ "answer": str, "session": str, "mode": str, "model": {"model": str, "provider": str} }`

## API Key (optional, opt‑in)

Set `ALITA_REQUIRE_API_KEY=true` to enforce API key checks. Provide keys via:
- `ALITA_API_KEY`: single key value
- `ALITA_API_KEYS`: comma‑separated values

Requests must include one of:
- `Authorization: Bearer <key>` (default header)
- Custom header via `ALITA_API_HEADER` (e.g., `X-API-Key: <key>`)
- Query parameter via `ALITA_API_QUERY_PARAM` (default `api_key`)

Example:
```
ALITA_REQUIRE_API_KEY=true \
ALITA_API_KEY=sk-demo-123 \
make run

curl -X POST http://127.0.0.1:8080/api/v1/query \
  -H "Authorization: Bearer sk-demo-123" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello"}'
```

Admin and limits:
- `ALITA_ADMIN_KEY`: admin bearer token for key management
- `ALITA_API_KEY_STORE`: path to key store (default `config/api_keys.json`)
- `ALITA_AUTH_OPEN_REG`: allow public key creation without admin (default `false`)
- `ALITA_RATE_LIMIT_ENABLED`: `true/false` (default `false`)
- `ALITA_RATE_LIMIT`: requests per window (default `60`)
- `ALITA_RATE_WINDOW`: window seconds (default `60`)

Key management endpoints (admin):
- `POST /api/v1/auth/keys` body `{ "owner": "you@example.com", "ttl_hours": 24 }` → returns `{ key_id, api_key }`
- `POST /api/v1/auth/keys/rotate` (present current key in Authorization) → returns new key
- `POST /api/v1/auth/keys/revoke` body `{ "key": "sk-..." }` or `{ "key_id": "..." }`
- `GET /api/v1/auth/keys/me` (present key) → returns info for current key
- `GET /api/v1/auth/keys` (admin) → list public metadata for keys

## CLI

- Configure once: `python -m tools.alita_cli configure --base-url http://127.0.0.1:8080 --api-key sk-demo-123`
- Run: `python -m tools.alita_cli query "What is the capital of France?"`
- Stream: `python -m tools.alita_cli query "Explain transformers" --stream`
- Output JSON: `python -m tools.alita_cli query "Hello" --output json --show-metadata`

Key management via CLI:
- Create: `python -m tools.alita_cli keys create you@example.com --ttl-hours 24 --admin-key admin`
- Rotate: `python -m tools.alita_cli keys rotate`
- Revoke: `python -m tools.alita_cli keys revoke --key sk-... --admin-key admin`
- Me: `python -m tools.alita_cli keys me`
- List: `python -m tools.alita_cli keys list --admin-key admin`

Env:
- `ALITA_BASE_URL` (default `http://127.0.0.1:8080`)
- `ALITA_API_KEY` (optional unless required by server)
- `ALITA_API_HEADER` (default `Authorization`)

## Notes
- This API gateway reuses the existing chat pipeline for consistency.
- `mode` and `max_tokens` are accepted for forward compatibility.
- No database or user management is introduced in this minimal layer.

## Redis Rate Limiting (Optional)

Enable multi-process rate limiting with Redis:

- Set env:
  - `ALITA_RATE_LIMIT_ENABLED=true`
  - `ALITA_RATE_LIMIT=60` (requests per window)
  - `ALITA_RATE_WINDOW=60` (seconds)
  - `ALITA_REDIS_URL=redis://redis:6379` (or your Redis URL)

Compose snippet:
```
version: '3.9'
services:
  alita:
    build: .
    environment:
      - ALITA_RATE_LIMIT_ENABLED=true
      - ALITA_RATE_LIMIT=60
      - ALITA_RATE_WINDOW=60
      - ALITA_REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```
The app auto-detects Redis when `ALITA_REDIS_URL` is set; otherwise it falls back to in-process limiting.
