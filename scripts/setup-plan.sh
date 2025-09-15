#!/usr/bin/env bash
# Setup plan and contract folders for a feature, outputting JSON.
set -euo pipefail
source "$(dirname "$0")/lib/sdd-common.sh"

if [[ $# -lt 1 ]]; then
  echo '{"ok":false,"error":"usage: setup-plan.sh specs/<slug>"}'
  exit 2
fi

SPEC_DIR="$1"
PLAN_FILE="${SPEC_DIR}/plan.md"
RESEARCH_FILE="${SPEC_DIR}/research.md"
CONTRACTS_DIR="${SPEC_DIR}/contracts"

mkdir -p "${CONTRACTS_DIR}"
[[ -f "${RESEARCH_FILE}" ]] || : > "${RESEARCH_FILE}"

printf '{'
printf '"ok":true,'
printf '"spec_dir":"%s",' "$SPEC_DIR"
printf '"plan_file":"%s",' "$PLAN_FILE"
printf '"research_file":"%s",' "$RESEARCH_FILE"
printf '"contracts_dir":"%s"' "$CONTRACTS_DIR"
printf '}\n'