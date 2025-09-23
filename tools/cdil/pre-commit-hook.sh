#!/bin/bash
# Pre-commit/push hook for Spec Kit + CDIL verification
# Prevents commits/pushes with spec or signature drift

set -e

echo "Running Spec Kit + CDIL pre-commit/push checks..."

# Check if spec files were modified
if git diff --cached --name-only | grep -E '\.(yaml|yml)$' | grep -q 'specs/'; then
    echo "Spec files detected in commit. Verifying spec lock..."
    
    # Run spec lock verification
    if ! node tools/spec-kit/verify-lock.js; then
        echo "❌ Spec lock verification failed. Please run Spec Kit compiler and commit updated lock file."
        exit 1
    fi
    echo "✅ Spec lock verification passed"
fi

# Check if source files were modified
if git diff --cached --name-only | grep -q -E 'src/.*\.(py|ts|js)$'; then
    echo "Source files detected in commit. Verifying signature lock..."
    
    # Run signature lock verification
    if ! bash tools/cdil/extract_all.sh --verify; then
        echo "❌ Signature lock verification failed. API surface has changed without updating lock file."
        exit 1
    fi
    echo "✅ Signature lock verification passed"
fi

echo "✅ All Spec Kit + CDIL checks passed"
exit 0