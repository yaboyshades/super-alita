#!/usr/bin/env bash
# check-task-prerequisites.sh - Validate required artifacts before task generation.
# Usage: ./scripts/check-task-prerequisites.sh [--json]

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/lib/sdd-common.sh
source "${SCRIPT_DIR}/lib/sdd-common.sh"

print_usage() {
    cat <<'USAGE'
Usage: ./scripts/check-task-prerequisites.sh [--json]
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
PRIMARY_SPEC="${FEATURE_DIR}/feature-spec.md"
LEGACY_SPEC="${FEATURE_DIR}/spec.md"
PRIMARY_PLAN="${FEATURE_DIR}/plan.md"
LEGACY_PLAN="${FEATURE_DIR}/implementation-plan.md"
RESEARCH="${FEATURE_DIR}/research.md"
DATA_MODEL="${FEATURE_DIR}/data-model.md"
QUICKSTART="${FEATURE_DIR}/quickstart.md"
CONTRACTS_DIR="${FEATURE_DIR}/contracts"

if [[ ! -d "${FEATURE_DIR}" ]]; then
    if ${JSON_MODE}; then
        missing_json=$(sdd_json_array "feature-spec.md" "plan.md")
        printf '{"ok":false,"missing":%s}\n' "${missing_json}"
    else
        printf 'ERROR: Feature directory not found: %s\n' "${FEATURE_DIR}"
        printf 'Run /specify first to create the feature structure.\n'
    fi
    exit 1
fi

failures=()
feature_spec_found=0
plan_found=0

if [[ -f "${PRIMARY_SPEC}" ]]; then
    feature_spec_found=1
elif [[ -f "${LEGACY_SPEC}" ]]; then
    feature_spec_found=1
else
    failures+=("feature-spec.md")
fi

if [[ -f "${PRIMARY_PLAN}" ]]; then
    plan_found=1
elif [[ -f "${LEGACY_PLAN}" ]]; then
    plan_found=1
else
    failures+=("plan.md")
fi

has_research=0
has_data_model=0
has_contracts=0
has_quickstart=0

if [[ -f "${RESEARCH}" ]]; then
    has_research=1
fi
if [[ -f "${DATA_MODEL}" ]]; then
    has_data_model=1
fi
if [[ -d "${CONTRACTS_DIR}" ]] && [[ -n "$(find "${CONTRACTS_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    has_contracts=1
fi
if [[ -f "${QUICKSTART}" ]]; then
    has_quickstart=1
fi

if ${JSON_MODE}; then
    ok_value="true"
    if ((${#failures[@]} > 0)); then
        ok_value="false"
    fi
    missing_json=$(sdd_json_array "${failures[@]}")
    printf '{"ok":%s,"missing":%s}\n' "${ok_value}" "${missing_json}"
else
    printf 'FEATURE_DIR:%s\n' "${FEATURE_DIR}"
    printf 'REQUIRED:\n'
    if ((feature_spec_found)); then
        printf '  \u2713 feature-spec.md\n'
    else
        printf '  \u2717 feature-spec.md\n'
    fi
    if ((plan_found)); then
        printf '  \u2713 plan.md\n'
    else
        printf '  \u2717 plan.md\n'
    fi

    printf '\nAVAILABLE_DOCS:\n'
    if ((has_research)); then
        printf '  \u2713 research.md\n'
    else
        printf '  \u2717 research.md\n'
    fi
    if ((has_data_model)); then
        printf '  \u2713 data-model.md\n'
    else
        printf '  \u2717 data-model.md\n'
    fi
    if ((has_contracts)); then
        printf '  \u2713 contracts/\n'
    else
        printf '  \u2717 contracts/\n'
    fi
    if ((has_quickstart)); then
        printf '  \u2713 quickstart.md\n'
    else
        printf '  \u2717 quickstart.md\n'
    fi

    if ((feature_spec_found == 0)); then
        printf 'ERROR: feature-spec.md not found in %s\n' "${FEATURE_DIR}"
        printf 'Run /specify first to create the feature structure.\n'
    fi
    if ((plan_found == 0)); then
        printf 'ERROR: plan.md not found in %s\n' "${FEATURE_DIR}"
        printf 'Run /plan first to create the plan.\n'
    fi
fi

if ((${#failures[@]} > 0)); then
    exit 1
fi
