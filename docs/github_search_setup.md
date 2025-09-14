# GitHub Search Integration — Setup and Usage

This repo already includes GitHub search abilities you can call from the runtime or via the generic ability execution endpoint. Adding a token raises rate limits and improves reliability.

## 1) Configure a token

1. Create a Personal Access Token (fine‑grained or classic both work).
   - Minimum scopes: public_repo or "Contents: Read-only"
2. Copy `.env.example` to `.env` and set:
   - `GITHUB_TOKEN=your_pat_here`
3. Restart your dev server if it's running.

Security notes:

- Never commit real tokens.
- Use a fine‑grained PAT when possible.

## 2) Abilities available

- `github_search_code` — search code snippets
  - Args: `q` (string), optional `language`, `repo`, `per_page` (<=50), `page` (<=10)
- `github_search_repos` — search repositories
  - Args: `q`, `sort` (stars|forks|updated), `order` (asc|desc), pagination
- `github_integration_spec` — code search + emits a structured integration plan from top results
  - Args: `q`, optional `language`, `repo`, pagination, `max_candidates` (<=10)

You can enumerate tools via the tool catalog: `GET /tools/catalog`.

## 3) Quick smoke test

Run the in‑process smoke test (no server required):

```powershell
# From repo root (after activating your venv)
python scripts/smoke_github_search.py
```

Expected output: HTTP 200, a handful of results, and basic fields printed.

## 4) Call from HTTP

When the API server is running:

```powershell
# Code search
curl -X POST "http://127.0.0.1:8080/ability/execute/github_search_code" `
  -H "Content-Type: application/json" `
  -d '{
    "args": { "q": "python parse csv", "per_page": 5 }
  }'

# Integration spec
curl -X POST "http://127.0.0.1:8080/ability/execute/github_integration_spec" `
  -H "Content-Type: application/json" `
  -d '{
    "args": { "q": "fastapi middleware", "language": "python" }
  }'
```

## 5) Integration guidance

- Prefer adopting existing, well‑maintained code over re‑implementing
- Vendor small snippets with license headers preserved; otherwise use as dependency
- Wrap with our public API, add tests, run ruff + mypy, and document usage

This aligns with the Library‑First and Simplicity constitutional articles.
