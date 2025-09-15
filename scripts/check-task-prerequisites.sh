#!/usr/bin/env bash
# Validate that spec/plan/tasks exist and have content. Block if not.
set -euo pipefail
source "$(dirname "$0")/lib/sdd-common.sh"

if [[ $# -lt 1 ]]; then
  echo '{"ok":false,"error":"usage: check-task-prerequisites.sh specs/<slug>"}'
  exit 2
fi

SPEC_DIR="$1"
REQUIRED_FILES=("feature-spec.md" "plan.md" "tasks.md")
missing=()

for file in "${REQUIRED_FILES[@]}"; do
  fpath="${SPEC_DIR}/${file}"
  if [[ ! -s "$fpath" ]]; then
    missing+=("$file")
  fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
  printf '{'
  printf '"ok":false,"missing":['
  for i in "${!missing[@]}"; do
    [[ $i -gt 0 ]] && printf ','
    printf '"%s"' "${missing[$i]}"
  done
  printf ']}\n'
  exit 1
else
  echo '{"ok":true}'
fi