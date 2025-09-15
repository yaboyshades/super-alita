#!/usr/bin/env bash
# setup-plan.sh - Prepare the implementation plan workspace for the branch.
# Usage: ./scripts/setup-plan.sh [--json]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/lib/sdd-common.sh
source "${SCRIPT_DIR}/lib/sdd-common.sh"

print_usage() {
    cat <<'USAGE'
Usage: ./scripts/setup-plan.sh [--json]
USAGE
}

JSON_MODE=false
for arg in "$@"; do
    case "$arg" in
        --json)
            JSON_MODE=true
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
    esac
done

REPO_ROOT=$(git rev-parse --show-toplevel)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if ! require_feature_branch "${CURRENT_BRANCH}"; then
    exit 1
fi

FEATURE_DIR="${REPO_ROOT}/specs/${CURRENT_BRANCH}"
FEATURE_SPEC="${FEATURE_DIR}/spec.md"
IMPL_PLAN="${FEATURE_DIR}/plan.md"

mkdir -p "${FEATURE_DIR}"

TEMPLATE="${REPO_ROOT}/templates/plan-template.md"
if [[ -f "${TEMPLATE}" ]]; then
    cp "${TEMPLATE}" "${IMPL_PLAN}"
fi

if ${JSON_MODE}; then
    printf '{"FEATURE_SPEC":"%s","IMPL_PLAN":"%s","SPECS_DIR":"%s","BRANCH":"%s"}\n' \
        "${FEATURE_SPEC}" "${IMPL_PLAN}" "${FEATURE_DIR}" "${CURRENT_BRANCH}"
else
    printf 'FEATURE_SPEC: %s\n' "${FEATURE_SPEC}"
    printf 'IMPL_PLAN: %s\n' "${IMPL_PLAN}"
    printf 'SPECS_DIR: %s\n' "${FEATURE_DIR}"
    printf 'BRANCH: %s\n' "${CURRENT_BRANCH}"
fi
