#!/usr/bin/env bash
set -euo pipefail

MAIN_BRANCH="${1:-master}"
UPSTREAM_REMOTE="${2:-upstream}"
DRY_RUN="${DRY_RUN:-false}"

echo "== Super Alita Repo Sync (POSIX) =="
if [[ "$DRY_RUN" == "true" ]]; then echo "(DryRun mode)"; fi

run() { echo "--> $*"; if [[ "$DRY_RUN" == "true" ]]; then return 0; fi; "$@"; }

run git fetch origin --prune || true
if git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
  run git fetch "$UPSTREAM_REMOTE" --prune || true
else
  echo "No upstream remote configured (ok)."
fi

run git checkout "$MAIN_BRANCH"
run git pull --ff-only origin "$MAIN_BRANCH" || run git pull origin "$MAIN_BRANCH"

if git rev-parse "$UPSTREAM_REMOTE/$MAIN_BRANCH" >/dev/null 2>&1; then
  echo "Merging $UPSTREAM_REMOTE/$MAIN_BRANCH..."
  if ! run git merge --ff-only "$UPSTREAM_REMOTE/$MAIN_BRANCH"; then
    run git merge --no-edit "$UPSTREAM_REMOTE/$MAIN_BRANCH" || echo "Manual conflict resolution required" >&2
  fi
fi

if [[ "$DRY_RUN" != "true" ]]; then
  if ! git diff --quiet; then
    git add -A
    git commit -m "chore(sync): manual sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" || echo "Nothing to commit"
    git push origin "$MAIN_BRANCH" || echo "Push failed" >&2
  else
    echo "No changes after sync."
  fi
fi

echo "Sync complete."
