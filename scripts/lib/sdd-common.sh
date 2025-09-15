#!/usr/bin/env bash
set -euo pipefail

# Common helper utilities for SDD shell tooling.
# Provides JSON logging and slug helpers that mirror the Python implementation.

log_json() {
    if [[ $# -lt 1 ]]; then
        echo "log_json requires a message argument" >&2
        return 1
    fi

    local message="${1}"
    shift

    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local python_bin
    if command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    elif command -v python >/dev/null 2>&1; then
        python_bin="python"
    else
        echo "log_json requires python3" >&2
        return 1
    fi

    "${python_bin}" - "${timestamp}" "${message}" "$@" <<'PY'
import json
import sys

timestamp, message, *pairs = sys.argv[1:]
data = {
    "timestamp": timestamp,
    "level": "info",
    "message": message,
}

message_override = None
for pair in pairs:
    if "=" not in pair:
        continue
    key, value = pair.split("=", 1)
    if key == "message":
        message_override = value
        continue
    data[key] = value

if message_override is not None:
    # Warn to stderr if both positional and key-value message are provided
    if message != message_override:
        print(
            f"WARNING: Overriding positional message '{message}' with key-value 'message={message_override}'",
            file=sys.stderr,
        )
    data["message"] = message_override
print(json.dumps(data, ensure_ascii=False))
PY
}

# Provides helpers shared across bash entrypoints that orchestrate
# specification-driven development workflows.
slugify() {
    local input="${1:-}"
    if [[ -z "${input}" ]]; then
        printf '\n'
        return 0
    fi

    local slug
    slug=$(printf '%s' "${input}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -e 's/[^a-z0-9][^a-z0-9]*/-/g' -e 's/^-*//' -e 's/-*$//')

    printf '%s\n' "${slug}"
}

# Require that the provided branch name follows the standard feature
# prefix pattern (three digits followed by a hyphenated slug). Emits a
# helpful error to stderr when the branch name does not match.
require_feature_branch() {
    local branch="${1:-}"

    if [[ -z "${branch}" ]]; then
        echo "ERROR: Branch name is required" >&2
        return 1
    fi

    if [[ ! "${branch}" =~ ^[0-9]{3}- ]]; then
        echo "ERROR: Not on a feature branch. Current branch: ${branch}" >&2
        echo "Feature branches should be named like: 001-feature-name" >&2
        return 1
    fi

    return 0
}

# Escape a string for safe inclusion inside a JSON string literal. Only
# covers the characters produced by our CLI tooling (ASCII + newlines).
sdd_json_escape() {
    local input="${1:-}"
    local length=${#input}
    local i char

    for ((i = 0; i < length; i++)); do
        char=${input:i:1}
        case "${char}" in
            '"') printf '\\"' ;;
            '\\') printf '\\\\' ;;
            $'\n') printf '\\n' ;;
            $'\r') printf '\\r' ;;
            $'\t') printf '\\t' ;;
            *) printf '%s' "${char}" ;;
        esac
    done
}

# Serialize the provided arguments into a compact JSON array. Values are
# escaped using the helper above to avoid introducing a jq dependency in
# lightweight shell scripts.
sdd_json_array() {
    if (($# == 0)); then
        printf '[]'
        return 0
    fi

    local first=1
    printf '['
    for element in "$@"; do
        if ((first)); then
            first=0
        else
            printf ','
        fi

        printf '"'
        sdd_json_escape "${element}"
        printf '"'
    done
    printf ']'
}

# Basic self-test when the script is executed directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    expected="my-feature-123"
    result="$(slugify "My Feature 123!")"

    if [[ "${result}" != "${expected}" ]]; then
        echo "slugify self-test failed: expected '${expected}', got '${result}'" >&2
        exit 1
    fi

    if command -v python3 >/dev/null 2>&1; then
        _python_for_tests="python3"
    elif command -v python >/dev/null 2>&1; then
        _python_for_tests="python"
    else
        echo "log_json self-test failed: python interpreter not found" >&2
        exit 1
    fi

    log_output="$(log_json "SDD helper smoke test" level=debug scope=slugify)"
    log_json_status=$?
    if [ $log_json_status -ne 0 ]; then
        echo "log_json failed with exit code $log_json_status" >&2
        exit $log_json_status
    fi
    if ! "${_python_for_tests}" - "$log_output" <<'PY'
import json
import sys

try:
    payload = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    raise SystemExit(f"log_json self-test failed: {exc}")

required_keys = {"timestamp", "level", "message"}
missing = required_keys.difference(payload)
if missing:
    raise SystemExit(f"log_json self-test failed: missing keys {sorted(missing)}")

if payload["message"] != "SDD helper smoke test":
    raise SystemExit(
        "log_json self-test failed: unexpected message field"
    )

if payload.get("level") != "debug":
    raise SystemExit(
        "log_json self-test failed: unexpected level field"
    )

if payload.get("scope") != "slugify":
    raise SystemExit(
        "log_json self-test failed: unexpected scope field"
    )
PY
    then
        exit 1
    fi

    echo "sdd-common self-tests passed"
fi
