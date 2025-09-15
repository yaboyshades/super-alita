#!/usr/bin/env bash
set -euo pipefail

OUT="docs/deep-dive"
ARTIFACT_DIR="$OUT/artifacts"
GRAPH_DIR="$OUT/graphs"
LOG_DIR="$OUT/logs"
mkdir -p "$ARTIFACT_DIR" "$GRAPH_DIR" "$LOG_DIR"

echo "== super-alita Deep Dive ==" | tee "$OUT/summary.txt"

# --- Repo inventory & git forensics ---
echo "[1/9] Repo inventory & git forensics..."
{ 
  echo "# Inventory"
  find . -path ./.git -prune -o -type f -print | sed 's|^\./||' | wc -l | awk '{print "Total files:",$1}'
  echo
  echo "## Top file types"
  find . -path ./.git -prune -o -type f -print0 \
  | xargs -0 -I{} sh -c 'printf "%s\n" "${1##*.}"' _ {} \
  | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -nr | head -n 20

  echo
  echo "## cloc (if available)"
  if command -v cloc >/dev/null 2>&1; then cloc --json . > "$OUT/artifacts/cloc.json" && echo "Saved cloc.json"; else echo "cloc not installed"; fi

  echo
  echo "## Git stats"
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 && {
    echo "- Contributors:"; git shortlog -sn | sed 's/^/  /'
    echo "- Churn (last 90 days):"; git log --since="90 days ago" --name-only --pretty=format: | grep -v '^$' | sort | uniq -c | sort -nr | head -n 30 | sed 's/^/  /'
  } || echo "Not a git repo"
} | tee "$OUT/artifacts/inventory.txt"

# --- Language detection ---
echo "[2/9] Language detection..."
LANG_HINTS=$(find . -path ./.git -prune -o -type f -regex '.*\.(js|ts|jsx|tsx|py|go|rs|toml|lock|json|yml|yaml)' -print | tr '\n' ' ')
HAS_JS=$(echo "$LANG_HINTS" | grep -E -q '\\.([jt]s|tsx?)|package\\.json|pnpm-lock|yarn\\.lock' && echo 1 || echo 0)
HAS_PY=$(echo "$LANG_HINTS" | grep -E -q '\\.py|requirements\\.txt|pyproject\\.toml|poetry\\.lock' && echo 1 || echo 0)
HAS_GO=$(echo "$LANG_HINTS" | grep -q 'go\\.mod' && echo 1 || echo 0)
HAS_RS=$(echo "$LANG_HINTS" | grep -q 'Cargo\\.toml' && echo 1 || echo 0)
echo "JS/TS=$HAS_JS PY=$HAS_PY GO=$HAS_GO RUST=$HAS_RS" | tee "$OUT/artifacts/langs.txt"

# --- JS/TS analysis ---
if [ "$HAS_JS" = "1" ]; then
  echo "[3/9] JS/TS checks..."
  if command -v node >/dev/null 2>&1; then
    PKG_MGR="npm"
    [ -f pnpm-lock.yaml ] && PKG_MGR="pnpm"
    [ -f yarn.lock ] && PKG_MGR="yarn"
    echo "Using $PKG_MGR" | tee -a "$LOG_DIR/js.log"

    # Install (no scripts for safety)
    if [ -f package.json ]; then
      case "$PKG_MGR" in
        npm) npm ci --ignore-scripts || npm install --ignore-scripts ;;
        yarn) yarn install --ignore-scripts || yarn install ;;
        pnpm) pnpm install --ignore-scripts || pnpm install ;;
      esac
    fi

    # Lint (if present and config exists)
    if [ -f node_modules/.bin/eslint ]; then
      if [ -f .eslintrc ] || [ -f .eslintrc.js ] || [ -f .eslintrc.json ] || [ -f .eslintrc.yaml ] || [ -f .eslintrc.yml ] || grep -q '"eslintConfig"' package.json 2>/dev/null; then
        ESLINT_OUT="$ARTIFACT_DIR/ts_lint.json"
        npx eslint . -f json | tee "$ESLINT_OUT" >/dev/null || true
        ESLINT_STATUS=${PIPESTATUS[0]}
        if [ "$ESLINT_STATUS" -ne 0 ] && [ ! -s "$ESLINT_OUT" ]; then
          printf '{"error":"eslint failed","exit_code":%s}\n' "$ESLINT_STATUS" > "$ESLINT_OUT"
        fi
      fi
    fi

    # Dep graph (dependency-cruiser if available)
    npx --yes dependency-cruiser@latest -T dot -x "node_modules|dist|build" . > "$GRAPH_DIR/dep.dot" 2>>"$LOG_DIR/js.log" || true

    # Vulnerabilities
    AUDIT_OUT="$ARTIFACT_DIR/npm_audit.json"
    DEPS_OUT="$ARTIFACT_DIR/npm_deps.json"
    case "$PKG_MGR" in
      npm)
        npm audit --json | tee "$AUDIT_OUT" >/dev/null || true
        AUDIT_STATUS=${PIPESTATUS[0]}
        if [ "$AUDIT_STATUS" -ne 0 ] && [ ! -s "$AUDIT_OUT" ]; then
          printf '{"error":"npm audit failed","exit_code":%s}\n' "$AUDIT_STATUS" > "$AUDIT_OUT"
        fi
        npm ls --json | tee "$DEPS_OUT" >/dev/null || true
        NPM_LS_STATUS=${PIPESTATUS[0]}
        if [ "$NPM_LS_STATUS" -ne 0 ] && [ ! -s "$DEPS_OUT" ]; then
          printf '{"error":"npm ls failed","exit_code":%s}\n' "$NPM_LS_STATUS" > "$DEPS_OUT"
        fi
        ;;
      yarn)
        yarn npm audit --json | tee "$AUDIT_OUT" >/dev/null || true
        YARN_AUDIT_STATUS=${PIPESTATUS[0]}
        if [ "$YARN_AUDIT_STATUS" -ne 0 ] && [ ! -s "$AUDIT_OUT" ]; then
          printf '{"error":"yarn npm audit failed","exit_code":%s}\n' "$YARN_AUDIT_STATUS" > "$AUDIT_OUT"
        fi
        yarn npm ls --json | tee "$DEPS_OUT" >/dev/null || true
        YARN_LS_STATUS=${PIPESTATUS[0]}
        if [ "$YARN_LS_STATUS" -ne 0 ] && [ ! -s "$DEPS_OUT" ]; then
          printf '{"error":"yarn npm ls failed","exit_code":%s}\n' "$YARN_LS_STATUS" > "$DEPS_OUT"
        fi
        ;;
      pnpm)
        pnpm audit --json | tee "$AUDIT_OUT" >/dev/null || true
        PNPM_AUDIT_STATUS=${PIPESTATUS[0]}
        if [ "$PNPM_AUDIT_STATUS" -ne 0 ] && [ ! -s "$AUDIT_OUT" ]; then
          printf '{"error":"pnpm audit failed","exit_code":%s}\n' "$PNPM_AUDIT_STATUS" > "$AUDIT_OUT"
        fi
        pnpm ls --json | tee "$DEPS_OUT" >/dev/null || true
        PNPM_LS_STATUS=${PIPESTATUS[0]}
        if [ "$PNPM_LS_STATUS" -ne 0 ] && [ ! -s "$DEPS_OUT" ]; then
          printf '{"error":"pnpm ls failed","exit_code":%s}\n' "$PNPM_LS_STATUS" > "$DEPS_OUT"
        fi
        ;;
    esac

    # Tests: collect (don’t run full suite yet)
    if [ -f node_modules/.bin/jest ]; then npx jest --listTests > "$OUT/artifacts/jest-tests.txt" || true; fi
    if [ -f node_modules/.bin/vitest ]; then npx vitest list > "$OUT/artifacts/vitest-tests.txt" || true; fi
  fi
fi

# --- Python analysis ---
if [ "$HAS_PY" = "1" ]; then
  echo "[4/9] Python checks..."
  PYBIN=$(command -v python3 || true)
  if [ -n "$PYBIN" ]; then
    # Create a virtual environment for isolation
    VENV_DIR="$OUT/venv"
    if [ ! -d "$VENV_DIR" ]; then
      "$PYBIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1090
    . "$VENV_DIR/bin/activate"
    PYBIN="$VENV_DIR/bin/python"
    "$PYBIN" -m pip install -q --upgrade pip
    "$PYBIN" -m pip install -q pip-audit pipdeptree flake8 flake8-json pytest || true
    # Check that flake8-json is installed
    if ! "$PYBIN" -c "import flake8_json" 2>/dev/null; then
      echo '{"error":"flake8-json plugin not installed"}' > "$ARTIFACT_DIR/python_lint.json"
    else
      "$VENV_DIR/bin/flake8" . --format=json | tee "$ARTIFACT_DIR/python_lint.json" >/dev/null || true
      PY_LINT_STATUS=${PIPESTATUS[0]}
      # Validate that output is valid JSON
      if ! "$PYBIN" -m json.tool < "$ARTIFACT_DIR/python_lint.json" >/dev/null 2>&1; then
        printf '{"error":"flake8 output is not valid JSON","exit_code":%s}\n' "$PY_LINT_STATUS" > "$ARTIFACT_DIR/python_lint.json"
      elif [ "$PY_LINT_STATUS" -ne 0 ] && [ ! -s "$ARTIFACT_DIR/python_lint.json" ]; then
        printf '{"error":"flake8 failed","exit_code":%s}\n' "$PY_LINT_STATUS" > "$ARTIFACT_DIR/python_lint.json"
      fi
    fi
    # Test discovery
    if [ -d tests ] || ls -1 *test*.py >/dev/null 2>&1; then
      "$PYBIN" -m pytest --collect-only -q > "$ARTIFACT_DIR/pytest-collect.txt" || true
    fi
  fi
fi

# --- Go analysis ---
if [ "$HAS_GO" = "1" ]; then
  echo "[5/9] Go checks..."
  if command -v go >/dev/null 2>&1; then
    go list -m all > "$OUT/artifacts/go-mods.txt" || true
    go vet ./... > "$OUT/artifacts/go-vet.txt" 2>&1 || true
    if command -v staticcheck >/dev/null 2>&1; then staticcheck ./... > "$OUT/artifacts/staticcheck.txt" || true; fi
    go test ./... -c >/dev/null 2>&1 || true
  fi
fi

# --- Rust analysis ---
if [ "$HAS_RS" = "1" ]; then
  echo "[6/9] Rust checks..."
  if command -v cargo >/dev/null 2>&1; then
    cargo tree -e features -J > "$OUT/artifacts/cargo-tree.json" || true
    cargo clippy --message-format=json > "$OUT/artifacts/cargo-clippy.json" 2>&1 || true
    if command -v cargo-audit >/dev/null 2>&1; then cargo audit -q -j > "$OUT/artifacts/cargo-audit.json" || true; fi
    cargo test --no-run >/dev/null 2>&1 || true
  fi
fi

# --- Secret scan (if available) ---
echo "[7/9] Secret scan..."
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect -v --report-format json --report-path "$OUT/artifacts/gitleaks.json" || true
else
  echo "gitleaks not installed" > "$OUT/artifacts/gitleaks.txt"
fi

# --- Container/Docker (optional) ---
echo "[8/9] Container scan..."
if command -v trivy >/dev/null 2>&1; then
  if [ -f Dockerfile ]; then trivy fs --format json --output "$OUT/artifacts/trivy-fs.json" . || true; fi
else
  echo "trivy not installed" > "$OUT/artifacts/trivy.txt"
fi

# --- SDD & Codecraft checklist synthesis (lightweight) ---
echo "[9/9] SDD heuristics..." 
{
  echo "## SDD Readiness Heuristics"
  echo "- Specs present?"; ls -1 | grep -E '^specs$|spec|docs' || true
  echo "- Contracts folder?"; find . -type d -name contracts | sed 's/^/  - /' || true
  echo "- Tests present?"; find . -type d -name tests | sed 's/^/  - /' || true
  echo "- CI workflows:"; ls -1 .github/workflows 2>/dev/null || echo "  (none)"
} > "$OUT/artifacts/sdd-readiness.txt"

echo "Deep dive artifacts -> $OUT"
