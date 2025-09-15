#!/usr/bin/env bash
set -euo pipefail

# serialize-json-array-examples.sh
#
# Purpose:
#   Demonstrate reliable strategies for serializing Bash arrays to JSON.
#
# Usage:
#   ./scripts/serialize-json-array-examples.sh
#
# Prerequisites:
#   - bash 4+
#   - jq (used for the reference implementation and validation)
#
# The script prints three sections:
#   1. jq --args       : direct conversion using jq's $ARGS.positional helper
#   2. printf join      : manual serialization with explicit JSON escaping
#   3. here-doc payload : embedding an array inside a larger JSON document
#
# Each output is validated with jq to ensure correct quoting and commas.

render_json_string() {
  local input=${1-}
  local length=${#input}
  local i char

  for (( i = 0; i < length; i++ )); do
    char=${input:i:1}
    case $char in
      '"') printf '\\"' ;;
      '\\') printf '\\\\' ;;
      $'\n') printf '\\n' ;;
      $'\r') printf '\\r' ;;
      $'\t') printf '\\t' ;;
      *) printf '%s' "$char" ;;
    esac
  done
}

build_with_jq() {
  jq -n '$ARGS.positional' --args "$@"
}

build_with_printf() {
  local elements=("$@")
  local first=1

  printf '['
  for element in "${elements[@]}"; do
    if (( first )); then
      first=0
    else
      printf ','
    fi

    printf '"'
    render_json_string "$element"
    printf '"'
  done
  printf ']\n'
}

build_here_doc() {
  local elements=("$@")
  local last_index=$(( ${#elements[@]} - 1 ))
  local lines=()

  for i in "${!elements[@]}"; do
    local escaped suffix
    escaped=$(render_json_string "${elements[$i]}")
    suffix=","
    if (( i == last_index )); then
      suffix=""
    fi
    lines+=("    \"${escaped}\"${suffix}")
  done

  local items_payload=""
  if ((${#lines[@]} > 0)); then
    items_payload=$(printf '%s\n' "${lines[@]}")
  fi
  cat <<JSON
{
  "items": [
${items_payload}
  ]
}
JSON
}

validate_json() {
  local label=$1
  local payload=$2

  if ! command -v jq >/dev/null 2>&1; then
    printf 'jq is required for validation.\n' >&2
    exit 1
  fi

  if jq -e . >/dev/null <<<"$payload"; then
    printf 'Validation (%s): OK\n' "$label"
  else
    printf 'Validation (%s): FAILED\n' "$label" >&2
    return 1
  fi
}

run_examples() {
  local sample=("$@")

  printf '%s\n' '--- jq --args example ---'
  local jq_json
  jq_json=$(build_with_jq "${sample[@]}")
  printf '%s\n\n' "$jq_json"

  printf '%s\n' '--- printf join example ---'
  local printf_json
  printf_json=$(build_with_printf "${sample[@]}")
  printf '%s\n\n' "$printf_json"

  printf '%s\n' '--- here-doc payload example ---'
  local doc
  doc=$(build_here_doc "${sample[@]}")
  printf '%s\n' "$doc"

  printf '\nValidation summary:\n'
  validate_json 'jq-array' "$jq_json"
  validate_json 'printf-array' "$printf_json"
  validate_json 'here-doc' "$doc"
}

main() {
  local sample_values=(
    "alpha"
    "beta gamma"
    "value with \"quotes\""
    $'line with\nnewline'
    'unicode ☃'
  )

  run_examples "${sample_values[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
