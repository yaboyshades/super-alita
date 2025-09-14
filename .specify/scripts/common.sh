#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
feature_branch_regex='^[0-9]{3}-'

die(){ echo "ERROR: $*" >&2; exit 1; }

ensure_feature_branch(){
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    [[ "$BRANCH" =~ $feature_branch_regex ]] || die "Not on feature branch (NNN-slug): $BRANCH"
}

abs(){ python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }

# Canonical feature paths (CMA v5.3 layout under features/)
feature_dir(){ echo "$ROOT/features/$(git rev-parse --abbrev-ref HEAD)"; }
spec_file(){ echo "$(feature_dir)/spec.md"; }
plan_file(){ echo "$(feature_dir)/plan.md"; }
tasks_file(){ echo "$(feature_dir)/tasks.md"; }