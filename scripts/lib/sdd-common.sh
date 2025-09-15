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

for pair in pairs:
    if "=" not in pair:
        continue
    key, value = pair.split("=", 1)
    if key == "message":
        continue
    data[key] = value

print(json.dumps(data, ensure_ascii=False))
PY
}

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
    if ! "${_python_for_tests}" - "$log_output" <<'PY'
import json
import sys

try:
    payload = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:  # pragma: no cover - defensive
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
