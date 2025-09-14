#!/usr/bin/env bash
set -euo pipefail
. ".specify/scripts/common.sh"

ensure_feature_branch

DIR="$(feature_dir)"
SPEC="$(spec_file)"
PLAN="$(plan_file)"
TASKS="$(tasks_file)"

jq -n \
	--arg dir "$(abs "$DIR")" \
	--arg spec "$(abs "$SPEC")" \
	--arg plan "$(abs "$PLAN")" \
	--arg tasks "$(abs "$TASKS")" \
	'{feature_dir:$dir,spec:$spec,plan:$plan,tasks:$tasks}'
#!/usr/bin/env bash
# Get paths for current feature branch without creating anything
# Used by commands that need to find existing feature files

set -e

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Get all paths
eval $(get_feature_paths)

# Check if on feature branch
check_feature_branch "$CURRENT_BRANCH" || exit 1

# Output paths (don't create anything)
echo "REPO_ROOT: $REPO_ROOT"
echo "BRANCH: $CURRENT_BRANCH"
echo "FEATURE_DIR: $FEATURE_DIR"
echo "FEATURE_SPEC: $FEATURE_SPEC"
echo "IMPL_PLAN: $IMPL_PLAN"
echo "TASKS: $TASKS"