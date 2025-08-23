# Secrets Hygiene

- Never commit secrets. Use env vars or secret stores.
- Copy `.env.example` to `.env` locally; do not commit `.env`.
- Pre-commit `detect-secrets` blocks common leaks.
- CI should provide secrets via the runner environment.

## Required
- GEMINI_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY
- REDIS_URL

## Optional
- SUPER_ALITA_MODE (shadow|act|batch)
- SUPER_ALITA_DATA_DIR
- SANDBOX_* limits


Use environment variables for all credentials.
