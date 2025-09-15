#!/usr/bin/env bash
# Create a new feature scaffold, outputting structured JSON for automation.
set -euo pipefail
source "$(dirname "$0")/lib/sdd-common.sh"

if [[ $# -lt 1 ]]; then
  echo '{"ok":false,"error":"usage: create-new-feature.sh \"Feature Name\""}'
  exit 2
fi

FEATURE_SLUG="$(slugify "$1")"
BRANCH_NAME="feat/${FEATURE_SLUG}"
SPEC_DIR="specs/${FEATURE_SLUG}"
SPEC_FILE="${SPEC_DIR}/feature-spec.md"
PLAN_FILE="${SPEC_DIR}/plan.md"
TASKS_FILE="${SPEC_DIR}/tasks.md"

mkdir -p "${SPEC_DIR}"

# Copy templates; avoid overwriting if already present
[[ -f "${SPEC_FILE}" ]] || cp templates/sdd/spec-template.md "${SPEC_FILE}"
[[ -f "${PLAN_FILE}" ]] || cp templates/sdd/plan-template.md "${PLAN_FILE}"
[[ -f "${TASKS_FILE}" ]] || cp templates/sdd/tasks-template.md "${TASKS_FILE}"

printf '{'
printf '"ok":true,'
printf '"feature_slug":"%s",' "$FEATURE_SLUG"
printf '"branch_name":"%s",' "$BRANCH_NAME"
printf '"spec_file":"%s",' "$SPEC_FILE"
printf '"plan_file":"%s",' "$PLAN_FILE"
printf '"tasks_file":"%s"' "$TASKS_FILE"
printf '}\n'