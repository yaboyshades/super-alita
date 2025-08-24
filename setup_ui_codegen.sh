#!/usr/bin/env bash
set -euo pipefail

cat > .env <<'EOF'
CODEGEN_ENABLED=true
GEMINI_MODEL=gemini-1.5-pro
# GEMINI_API_KEY=your-gemini-key
CODEGEN_MINIMAL=0
PROMPT_VERSION=2.1.0
SCHEMA_VERSION=alg_extraction_v1.2
REDIS_URL=redis://localhost:6379
EOF
