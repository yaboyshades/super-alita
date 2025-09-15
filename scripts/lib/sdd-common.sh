# Common SDD shell functions
set -euo pipefail

slugify() {
  # Lowercase, replace non-alnum with '-', and collapse repeats (POSIX safe)
  echo "$*" \
    | tr '[:upper:]' '[:lower:]' \
    | tr -c 'a-z0-9' '-' \
    | sed 's/--*/-/g; s/^-//; s/-$//'
}

# Structured logging (JSON)
log_json() {
  local level="${1:-info}"
  local component="${2:-sdd}"
  local msg="${3:-}"
  local ts
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  printf '{"ts":"%s","level":"%s","component":"%s","msg":"%s"}\n' "$ts" "$level" "$component" "$msg" >&2
}