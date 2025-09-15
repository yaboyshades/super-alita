#!/usr/bin/env bash
# create-new-feature.sh - Bootstrap a new feature branch and spec template.
# Usage: ./scripts/create-new-feature.sh [--json] <feature description>

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/lib/sdd-common.sh
source "${SCRIPT_DIR}/lib/sdd-common.sh"

print_usage() {
    cat <<'USAGE'
Usage: ./scripts/create-new-feature.sh [--json] <feature description>
USAGE
}

JSON_MODE=false
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --json)
            JSON_MODE=true
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

if ((${#ARGS[@]} == 0)); then
    print_usage >&2
    exit 1
fi

FEATURE_DESCRIPTION="${ARGS[*]}"
REPO_ROOT=$(git rev-parse --show-toplevel)
SPECS_DIR="${REPO_ROOT}/specs"
mkdir -p "${SPECS_DIR}"

HIGHEST=0
if compgen -G "${SPECS_DIR}/*" >/dev/null; then
    for entry in "${SPECS_DIR}"/*; do
        [[ -d "${entry}" ]] || continue
        basename=$(basename "${entry}")
        if [[ "${basename}" =~ ^([0-9]+) ]]; then
            number=${BASH_REMATCH[1]}
            number=$((10#${number}))
            if ((number > HIGHEST)); then
                HIGHEST=${number}
            fi
        fi
    done
fi

NEXT=$((HIGHEST + 1))
FEATURE_NUM=$(printf "%03d" "${NEXT}")

SLUG=$(slugify "${FEATURE_DESCRIPTION}")
if [[ -z "${SLUG}" ]]; then
    SLUG="feature"
fi

IFS='-' read -r -a SLUG_PARTS <<<"${SLUG}"
SELECTED=()
for part in "${SLUG_PARTS[@]}"; do
    [[ -z "${part}" ]] && continue
    SELECTED+=("${part}")
    if ((${#SELECTED[@]} == 3)); then
        break
    fi
done

if ((${#SELECTED[@]} == 0)); then
    SELECTED=("feature")
fi

BRANCH_SUFFIX=$(IFS=-; echo "${SELECTED[*]}")
BRANCH_NAME="${FEATURE_NUM}-${BRANCH_SUFFIX}"

git checkout -b "${BRANCH_NAME}"

FEATURE_DIR="${SPECS_DIR}/${BRANCH_NAME}"
mkdir -p "${FEATURE_DIR}"

TEMPLATE="${REPO_ROOT}/templates/spec-template.md"
SPEC_FILE="${FEATURE_DIR}/spec.md"

if [[ -f "${TEMPLATE}" ]]; then
    cp "${TEMPLATE}" "${SPEC_FILE}"
else
    printf 'Warning: Template not found at %s\n' "${TEMPLATE}" >&2
    : >"${SPEC_FILE}"
fi

if ${JSON_MODE}; then
    printf '{"BRANCH_NAME":"%s","SPEC_FILE":"%s","FEATURE_NUM":"%s"}\n' \
        "${BRANCH_NAME}" "${SPEC_FILE}" "${FEATURE_NUM}"
else
    printf 'BRANCH_NAME: %s\n' "${BRANCH_NAME}"
    printf 'SPEC_FILE: %s\n' "${SPEC_FILE}"
    printf 'FEATURE_NUM: %s\n' "${FEATURE_NUM}"
fi
