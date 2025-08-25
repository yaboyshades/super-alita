#!/usr/bin/env bash
set -euo pipefail

# Load local env if present
if [[ -f .env ]]; then
  set -a && . ./.env && set +a
fi

# Create .env from example if missing
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "PYTHONPATH=./src" >> .env
fi

# Never print API keys; just append if present in agent env
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
  grep -q '^GEMINI_API_KEY=' .env || echo "GEMINI_API_KEY=${GEMINI_API_KEY}" >> .env
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  grep -q '^OPENAI_API_KEY=' .env || echo "OPENAI_API_KEY=${OPENAI_API_KEY}" >> .env
fi
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  grep -q '^ANTHROPIC_API_KEY=' .env || echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" >> .env
fi

python -m pip install -U pip
pip install -r requirements.txt -r requirements-test.txt

# Optional: pre-commit normalize (if installed)
if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install || true
  pre-commit run --all-files || true
fi

echo "Setup complete."
