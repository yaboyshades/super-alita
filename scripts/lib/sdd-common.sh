#!/usr/bin/env bash
set -euo pipefail

# Common helper utilities for SDD shell tooling.
# Provides a slugify helper that mirrors the Python implementation.
slugify() {
    local input="${1:-}"
    if [[ -z "${input}" ]]; then
        printf '\n'
        return 0
    fi

    local slug
    slug=$(printf '%s' "${input}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/-/g' \
        | sed -E 's/^-+|-+$//g')

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

    echo "slugify self-test passed"
fi
