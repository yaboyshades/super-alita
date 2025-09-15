#!/usr/bin/env bash
set -euo pipefail

# Common helper utilities for SDD shell tooling.
# Provides helpers shared across bash entrypoints that orchestrate
# specification-driven development workflows.
slugify() {
    local input="${1:-}"
    if [[ -z "${input}" ]]; then
        printf '\n'
        return 0
    fi

    local transliterated
    if command -v iconv >/dev/null 2>&1; then
        transliterated=$(printf '%s' "${input}" | iconv -f utf-8 -t ascii//TRANSLIT 2>/dev/null)
        if [[ $? -ne 0 || -z "${transliterated}" ]]; then
            transliterated="${input}"
        fi
    else
        transliterated=$(python3 - <<'PY' "${input}"
import sys
import unicodedata


def _transliterate(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    fallback_map = {
        "ß": "ss",
        "Ø": "O",
        "ø": "o",
        "Đ": "D",
        "đ": "d",
        "Ł": "L",
        "ł": "l",
        "Æ": "AE",
        "æ": "ae",
        "Œ": "OE",
        "œ": "oe",
        "Þ": "Th",
        "þ": "th",
    }
    without_marks = []
    for ch in normalized:
        if unicodedata.category(ch).startswith("M"):
            continue
        without_marks.append(fallback_map.get(ch, ch))
    ascii_text = "".join(without_marks)
    return ascii_text.encode("ascii", "ignore").decode("ascii")


print(_transliterate(sys.argv[1]), end="")
PY
        )
    fi

    local slug
    slug=$(printf '%s' "${transliterated}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/-/g' \
        | sed -E 's/^-+|-+$//g')

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

    echo "slugify self-test passed"
fi
