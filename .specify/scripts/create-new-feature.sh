#!/usr/bin/env bash
set -euo pipefail
. ".specify/scripts/common.sh"

SLUG="${1:-feature}"

# Determine next sequence
LAST=$(git for-each-ref --format='%(refname:short)' refs/heads | awk -F- '/^[0-9]{3}-/{print $1}' | sort -n | tail -1)
NUM=$(( ${LAST:-0} + 1 ))

BRANCH=$(printf '%03d-%s' "$NUM" "$SLUG" | tr ' ' '-')
git checkout -b "$BRANCH"

DIR="features/$BRANCH"
mkdir -p "$DIR"

: > "$DIR/spec.md"

# JSON output with absolute paths per spec
abspath() { python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }

jq -n \
    --arg b "$BRANCH" \
    --arg d "$(abspath "$DIR")" \
    --arg s "$(abspath "$DIR/spec.md")" \
    --arg n "$NUM" \
    '{BRANCH_NAME:$b,FEATURE_DIR:$d,SPEC_FILE:$s,FEATURE_NUM:$n}'
#!/usr/bin/env bash
# Create a new feature with branch, directory structure, and template
# Usage: ./create-new-feature.sh "feature description"
#        ./create-new-feature.sh --json "feature description"

set -e

JSON_MODE=false

# Collect non-flag args
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --json)
            JSON_MODE=true
            ;;
        --help|-h)
            echo "Usage: $0 [--json] <feature_description>"; exit 0 ;;
        *)
            ARGS+=("$arg") ;;
    esac
done

FEATURE_DESCRIPTION="${ARGS[*]}"
if [ -z "$FEATURE_DESCRIPTION" ]; then
        echo "Usage: $0 [--json] <feature_description>" >&2
        exit 1
fi

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel)
SPECS_DIR="$REPO_ROOT/specs"

# Create specs directory if it doesn't exist
mkdir -p "$SPECS_DIR"

# Find the highest numbered feature directory
HIGHEST=0
if [ -d "$SPECS_DIR" ]; then
    for dir in "$SPECS_DIR"/*; do
        if [ -d "$dir" ]; then
            dirname=$(basename "$dir")
            number=$(echo "$dirname" | grep -o '^[0-9]\+' || echo "0")
            number=$((10#$number))
            if [ "$number" -gt "$HIGHEST" ]; then
                HIGHEST=$number
            fi
        fi
    done
fi

# Generate next feature number with zero padding
NEXT=$((HIGHEST + 1))
FEATURE_NUM=$(printf "%03d" "$NEXT")

# Create branch name from description
BRANCH_NAME=$(echo "$FEATURE_DESCRIPTION" | \
    tr '[:upper:]' '[:lower:]' | \
    sed 's/[^a-z0-9]/-/g' | \
    sed 's/-\+/-/g' | \
    sed 's/^-//' | \
    sed 's/-$//')

# Extract 2-3 meaningful words
WORDS=$(echo "$BRANCH_NAME" | tr '-' '\n' | grep -v '^$' | head -3 | tr '\n' '-' | sed 's/-$//')

# Final branch name
BRANCH_NAME="${FEATURE_NUM}-${WORDS}"

# Create and switch to new branch
git checkout -b "$BRANCH_NAME"

# Create feature directory
FEATURE_DIR="$SPECS_DIR/$BRANCH_NAME"
mkdir -p "$FEATURE_DIR"

# Copy template if it exists
TEMPLATE="$REPO_ROOT/templates/spec-template.md"
SPEC_FILE="$FEATURE_DIR/spec.md"

if [ -f "$TEMPLATE" ]; then
    cp "$TEMPLATE" "$SPEC_FILE"
else
    echo "Warning: Template not found at $TEMPLATE" >&2
    touch "$SPEC_FILE"
fi

if $JSON_MODE; then
    printf '{"BRANCH_NAME":"%s","SPEC_FILE":"%s","FEATURE_NUM":"%s"}\n' \
        "$BRANCH_NAME" "$SPEC_FILE" "$FEATURE_NUM"
else
    # Output results for the LLM to use (legacy key: value format)
    echo "BRANCH_NAME: $BRANCH_NAME"
    echo "SPEC_FILE: $SPEC_FILE"
    echo "FEATURE_NUM: $FEATURE_NUM"
fi