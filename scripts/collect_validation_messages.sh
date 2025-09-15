#!/usr/bin/env bash
#
# Collect repository validation messages and emit JSON for downstream tooling.
#
# Usage:
#   bash scripts/collect_validation_messages.sh
#
# Verification steps:
#   1. Run the script from the repository root.
#   2. Ensure the printed JSON payload contains at least three entries.
#   3. Confirm each message string is distinct so consumers can render
#      actionable feedback.
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

validation_messages=()
error_count=0

add_message() {
  local status=$1
  local message=$2
  validation_messages+=("${status}::${message}")
  if [[ ${status} == "error" ]]; then
    ((error_count++))
  fi
}

# Check: streaming orchestrator module present
if [[ -f "src/reug_runtime/router.py" ]]; then
  add_message "info" "reug_runtime/router.py present for streaming orchestration"
else
  add_message "error" "Missing streaming orchestrator module src/reug_runtime/router.py"
fi

# Check: router parser still recognises tool_call tags
if grep -q "<tool_call>" "src/reug_runtime/router.py"; then
  add_message "info" "Streaming router parses <tool_call> tags"
else
  add_message "warning" "Streaming router missing <tool_call> parser"
fi

# Check: runtime tests cover streaming smoke path
if [[ -f "tests/runtime/test_router_smoke.py" ]]; then
  add_message "info" "tests/runtime/test_router_smoke.py available for regression coverage"
else
  add_message "error" "Missing runtime smoke test for router"
fi

# Check: PATCHMAP documents streaming orchestrator behaviour
if grep -iq "streaming orchestrator" "PATCHMAP.md"; then
  add_message "info" "PATCHMAP.md documents the streaming orchestrator contract"
else
  add_message "warning" "PATCHMAP.md missing streaming orchestrator documentation"
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "Python interpreter not found" >&2
    exit 1
  fi
fi

JSON_OUTPUT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/validation_messages_to_json.py" "${validation_messages[@]}")
PYTHON_EXIT_CODE=$?
if [[ ${PYTHON_EXIT_CODE} -ne 0 ]]; then
  echo "Error: Python script validation_messages_to_json.py failed with exit code ${PYTHON_EXIT_CODE}" >&2
  exit 1
fi
printf '%s\n' "${JSON_OUTPUT}"
exit ${error_count}
