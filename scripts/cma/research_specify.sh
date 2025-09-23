#!/usr/bin/env bash
set -euo pipefail

print_usage() {
    cat <<'USAGE'
Usage: scripts/cma/research_specify.sh "High level description"
USAGE
}

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

DESCRIPTION="$*"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
CREATE_SCRIPT="${REPO_ROOT}/scripts/create-new-feature.sh"
if [[ ! -x "${CREATE_SCRIPT}" ]]; then
    echo "missing create-new-feature.sh" >&2
    exit 1
fi

RAW_OUTPUT=$("${CREATE_SCRIPT}" --json "${DESCRIPTION}")
BRANCH=$(printf '%s' "${RAW_OUTPUT}" | python -c 'import sys,json; data=json.load(sys.stdin); print(data["BRANCH_NAME"])')
SPEC=$(printf '%s' "${RAW_OUTPUT}" | python -c 'import sys,json; data=json.load(sys.stdin); print(data["SPEC_FILE"])')

cp "${REPO_ROOT}/templates/research-agent-spec-template.md" "${SPEC}"
{
    echo "# Input"
    echo "${DESCRIPTION}"
    echo ""
    cat "${SPEC}"
} > "${SPEC}.tmp"
mv "${SPEC}.tmp" "${SPEC}"

echo "{"branch":"${BRANCH}","spec":"${SPEC}"}"
