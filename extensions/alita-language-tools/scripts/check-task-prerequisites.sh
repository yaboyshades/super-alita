#!/usr/bin/env bash
# Check that required planning artifacts exist for the current feature branch
# Usage: ./check-task-prerequisites.sh [--json]

set -euo pipefail

JSON_MODE=false
for arg in "$@"; do
    case "$arg" in
        --json) JSON_MODE=true ;;
        --help|-h) echo "Usage: $0 [--json]"; exit 0 ;;
    esac
done

build_json_array() {
    if (( $# == 0 )); then
        echo "[]"
    else
        printf '%s
' "$@" | jq -R . | jq -s
    fi
}

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/common.sh"

# Get all paths
eval $(get_feature_paths)

# Check if on feature branch
check_feature_branch "$CURRENT_BRANCH" || exit 1

# Ensure feature directory exists
if [[ ! -d "$FEATURE_DIR" ]]; then
    if $JSON_MODE; then
        missing_json=$(build_json_array "feature-spec.md" "plan.md")
        printf '{"ok":false,"missing":%s}\n' "$missing_json"
    else
        echo "ERROR: Feature directory not found: $FEATURE_DIR"
        echo "Run /specify first to create the feature structure."
    fi
    exit 1
fi

failures=()

feature_spec_label="feature-spec.md"
feature_spec_candidates=("$FEATURE_DIR/$feature_spec_label" "$FEATURE_DIR/spec.md")
feature_spec_found=0
for candidate in "${feature_spec_candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
        feature_spec_found=1
        FEATURE_SPEC="$candidate"
        break
    fi
done
if (( feature_spec_found == 0 )); then
    failures+=("$feature_spec_label")
fi

plan_label="plan.md"
plan_candidates=("$FEATURE_DIR/$plan_label" "$FEATURE_DIR/implementation-plan.md")
plan_found=0
for candidate in "${plan_candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
        plan_found=1
        IMPL_PLAN="$candidate"
        break
    fi
done
if (( plan_found == 0 )); then
    failures+=("$plan_label")
fi

# Optional design documents
has_research=0
has_data_model=0
has_contracts=0
has_quickstart=0
if [[ -f "$RESEARCH" ]]; then
    has_research=1
fi
if [[ -f "$DATA_MODEL" ]]; then
    has_data_model=1
fi
if [[ -d "$CONTRACTS_DIR" ]] && [[ -n "$(ls -A "$CONTRACTS_DIR" 2>/dev/null)" ]]; then
    has_contracts=1
fi
if [[ -f "$QUICKSTART" ]]; then
    has_quickstart=1
fi

if $JSON_MODE; then
    ok_value="true"
    if (( ${#failures[@]} > 0 )); then
        ok_value="false"
    fi
    missing_json=$(build_json_array "${failures[@]}")
    printf '{"ok":%s,"missing":%s}
' "$ok_value" "$missing_json"
else
    echo "FEATURE_DIR:$FEATURE_DIR"
    echo "REQUIRED:"
    if (( feature_spec_found )); then
        echo "  ✓ feature-spec.md"
    else
        echo "  ✗ feature-spec.md"
    fi
    if (( plan_found )); then
        echo "  ✓ plan.md"
    else
        echo "  ✗ plan.md"
    fi

    echo "AVAILABLE_DOCS:"
    if (( has_research )); then
        echo "  ✓ research.md"
    else
        echo "  ✗ research.md"
    fi
    if (( has_data_model )); then
        echo "  ✓ data-model.md"
    else
        echo "  ✗ data-model.md"
    fi
    if (( has_contracts )); then
        echo "  ✓ contracts/"
    else
        echo "  ✗ contracts/"
    fi
    if (( has_quickstart )); then
        echo "  ✓ quickstart.md"
    else
        echo "  ✗ quickstart.md"
    fi

    if (( feature_spec_found == 0 )); then
        echo "ERROR: feature-spec.md not found in $FEATURE_DIR"
        echo "Run /specify first to create the feature structure."
    fi
    if (( plan_found == 0 )); then
        echo "ERROR: plan.md not found in $FEATURE_DIR"
        echo "Run /plan first to create the plan."
    fi
fi

if (( ${#failures[@]} > 0 )); then
    exit 1
fi
