#!/usr/bin/env bash
set -euo pipefail

OUT=".github/copilot-instructions.md"
mkdir -p .github

# --- Dependencies & guards ---
check_dependency() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: Required command '$1' not found. $2" >&2
    exit 1
  fi
}
check_dependency "rg" "Install ripgrep: https://github.com/BurntSushi/ripgrep"
[ -f "package.json" ] && check_dependency "jq" "Install jq: https://stedolan.github.io/jq/download/"

# Safer globbing
shopt -s nullglob

h() { echo "$1" >> "$OUT"; }
code() { printf '\n```%s\n' "$1" >> "$OUT"; cat >> "$OUT"; printf '```\n' >> "$OUT"; }

# Reset output file
: > "$OUT"

h "# High-Impact Context"
h "Generated on: $(date -Iseconds)"
h "Repo: $(basename "$PWD")"

# Section 0: AI Agent Primer & Persona
h "\n## 0) AI Agent Primer & Persona"
cat >> "$OUT" <<'EOPRIMER'
- You are a senior engineer deeply familiar with this repository, its conventions, and its quirks.
- Ground answers in real files/paths/exports; prefer edits to existing modules when they match patterns.
- Be concise but precise: provide copy-pasteable code for common tasks; for edge cases, explain and point to exemplars.
- Never hallucinate APIs. If uncertain, instruct the user to open the referenced file or propose a safe alternative.
- **Context triggers:**
  - If asked about **auth** → see `## Failure Mode Library` and `lib/auth.*`.
  - If asked about **adding an API** → see `## Common Tasks (Recipes)`.
  - If asked about **testing** → see `## Developer Workflows` and `## Project Conventions`.
EOPRIMER

h "\n## Architecture Overview"

compose_files=(**/docker-compose*.yml)
if (( ${#compose_files[@]} )); then
  h "Found docker-compose files:"
  printf '%s\n' "${compose_files[@]}" | sed 's/^/- /' >> "$OUT"
fi

dockerfiles=(**/Dockerfile*)
if (( ${#dockerfiles[@]} )); then
  h "Dockerfiles found:"
  printf '%s\n' "${dockerfiles[@]}" | sed 's/^/- /' >> "$OUT"
fi

h "\n## Developer Workflows"

if [ -f package.json ]; then
  h "Node scripts:"
  jq -r '.scripts' package.json | sed 's/^/    /' >> "$OUT" || true
fi

if [ -f pyproject.toml ]; then
  h "Python tasks (pyproject):"
  rg -n "\\[(tool\\.|project\\.)" pyproject.toml | sed 's/^/    /' >> "$OUT" || true
fi

for mf in Makefile justfile; do
  [ -f "$mf" ] && { h "\n$mf targets:"; rg -n "^[a-zA-Z0-9_.-]+:([^=]|$)" "$mf" | sed 's/^/    /' >> "$OUT"; } || true
done

h "\n## Project Conventions"

rg -n "eslint|prettier|tsconfig" --hidden -S | sed 's/^/- /' | head -50 >> "$OUT" || true
rg -n "ruff|flake8|black|mypy|pyright" --hidden -S | sed 's/^/- /' | head -50 >> "$OUT" || true
rg -n "gofmt|golangci-lint|clippy|rustfmt" --hidden -S | sed 's/^/- /' | head -50 >> "$OUT" || true

h "\n## Configuration & Secrets"
h "### Environment Variables"

env_patterns=("process\\.env" "dotenv" "os\\.environ" "VITE_" "NEXT_PUBLIC_" "REACT_APP_")
for pattern in "${env_patterns[@]}"; do
  rg -n --no-heading "$pattern" -g '!**/node_modules/**' -g '!**/dist/**' -g '!**/build/**' -S | head -20 >> "$OUT" 2>/dev/null || true
done

h "### Environment Files"
for f in .env .env.example .env.local .env.development; do
  if [ -f "$f" ]; then
    h "Found: $f"
    h "Sample keys:"
    grep -E "^[A-Z_]+=" "$f" | sed 's/=.*$//' | head -5 | sed 's/^/    /' >> "$OUT" || true
  fi
done

h "\n## Module & Service Map"
rg -n "^\s*(app|main|server|router|routes)\.(ts|js|py|go)|@app\.get|express\(|FastAPI\(|Flask\(|gorilla/mux" -S | sed 's/^/- /' | head -80 >> "$OUT" || true

h "\n## Common Tasks (Recipes)"
h "### Recipe: New API Endpoint (example)"
code sh <<'EOF'
# Python FastAPI example
uv run fastapi dev app.py  # or: uvicorn app:app --reload
# Add file: services/api/routers/widgets.py with router + test in tests/api/test_widgets.py
pytest -q
EOF

h "### Recipe: New React Component (example)"
code sh <<'EOF'
# React (Vite)
pnpm run dev
# Add: apps/web/src/components/Button.tsx and export from index.ts
pnpm run test -w web
EOF

h "### Recipe: New Database Migration (example)"
code sh <<'EOF'
# Python Alembic
alembic revision -m "add widgets"
alembic upgrade head
pytest -m db -q
EOF

h "\n## Failure Mode Library"
cat >> "$OUT" <<'EOL'
- **Module not found: `@/components`** → Ensure workspace hoists; run package manager in the affected app (e.g., `pnpm -w install`).
- **`node-gyp` build fails** → Requires Python 3.9 and build tools; see scripts/bootstrap-build.md.
- **Pytest timeouts in CI** → Increase `--timeout`/`-k` scope; review parallelism and DB setup.
EOL

h "\n## Actually…"
cat >> "$OUT" <<'EOL'
- We skip full Docker locally; use `pnpm -w dev` and `nox -s run-api`.
- Frontend default port is **3001**, not 3000, due to reverse proxy.
- Hotfixes commit directly to `main` with `hotfix:` prefix; PR after release.
EOL

h "\n## Non-Code Assets & Documentation"
if [ -d "public" ] || [ -d "static" ] || [ -d "assets" ]; then
  h "### Static Assets"
  find public static assets -type f \( -name "*.ico" -o -name "*.svg" -o -name "*.png" -o -name "*.jpg" \) 2>/dev/null | head -10 | sed 's/^/- /' >> "$OUT" || true
fi

if [ -d "docs" ] || [ -d "architecture" ] || [ -d "adr" ]; then
  h "### Key Documentation"
  find docs architecture adr -name "*.md" -o -name "*.mdx" 2>/dev/null | head -10 | sed 's/^/- /' >> "$OUT" || true
fi

h "\n## Glossary & Acronyms"
cat >> "$OUT" <<'EOL'
- ADR: Architectural Decision Record
- E2E: End-to-End tests
- OTel: OpenTelemetry
EOL

echo "Wrote $OUT"
