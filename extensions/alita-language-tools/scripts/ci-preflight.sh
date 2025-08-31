#!/usr/bin/env bash
set -euo pipefail

if ! command -v wasm-tools >/dev/null 2>&1; then
  echo "wasm-tools not found" >&2; exit 2
fi

if [[ "${REQUIRE_REAL_WIT:-0}" == "1" ]]; then
  META_FILE="src/generated/.codegen.meta.json"
  if [[ ! -f "$META_FILE" ]]; then echo "Missing $META_FILE" >&2; exit 3; fi
  COUNT=$(jq '.components | length' "$META_FILE" 2>/dev/null || echo 0)
  if [[ "$COUNT" -eq 0 ]]; then echo "REQUIRE_REAL_WIT=1 but meta shows 0 components" >&2; exit 4; fi
fi

echo "CI preflight OK"
