#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
constitutional-gates.sh --spec <path> --plan <path>

Validate SDD artifacts against constitutional gate requirements.

Options:
  --spec PATH   Path to the specification markdown or text file
  --plan PATH   Path to the implementation plan markdown or text file
  -h, --help    Show this message and exit

Dependencies:
  jq            Command-line JSON processor (https://stedolan.github.io/jq/)
USAGE
}

if (($# > 0)); then
    for arg in "$@"; do
        if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
            usage
            exit 0
        fi
    done
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required by constitutional-gates.sh but was not found in PATH." >&2
    echo "Install jq from https://stedolan.github.io/jq/ and retry." >&2
    exit 1
fi

spec_path=""
plan_path=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --spec)
            if [[ $# -lt 2 ]]; then
                echo "error: --spec requires a path" >&2
                usage
                exit 1
            fi
            spec_path="$2"
            shift 2
            ;;
        --plan)
            if [[ $# -lt 2 ]]; then
                echo "error: --plan requires a path" >&2
                usage
                exit 1
            fi
            plan_path="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$spec_path" || -z "$plan_path" ]]; then
    echo "error: both --spec and --plan must be provided" >&2
    usage
    exit 1
fi

if [[ ! -f "$spec_path" ]]; then
    echo "error: spec file not found: $spec_path" >&2
    exit 1
fi

if [[ ! -f "$plan_path" ]]; then
    echo "error: plan file not found: $plan_path" >&2
    exit 1
fi

messages=()

if ! grep -qi "feature[[:space:]]*id" "$spec_path"; then
    messages+=("Spec missing Feature ID (Article II)")
fi

if ! grep -Eiq "definition[[:space:]]+of[[:space:]]+done|\\bDoD\\b" "$plan_path"; then
    messages+=("Plan missing DoD (Article XV)")
fi

ok_literal="true"
if ((${#messages[@]} > 0)); then
    ok_literal="false"
fi

jq_args=()
jq_messages=()

if ((${#messages[@]} > 0)); then
    for idx in "${!messages[@]}"; do
        key="msg${idx}"
        jq_args+=(--arg "$key" "${messages[$idx]}")
        jq_messages+=("\$${key}")
    done
fi

message_body=""
if ((${#jq_messages[@]} > 0)); then
    message_body=$(printf ',%s' "${jq_messages[@]}")
    message_body="${message_body:1}"
fi

jq_filter="{\"ok\":${ok_literal},\"messages\":[${message_body}]}"

jq -cn "${jq_args[@]:-}" "$jq_filter"
