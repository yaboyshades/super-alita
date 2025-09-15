# Check SDD artifacts for Constitutional compliance (Articles II, XV-XX)
# Usage: constitutional_check <spec_dir>
set -euo pipefail

constitutional_check() {
  local spec_dir="$1"
  local ok=true
  local messages=()

  # Article II: Spec must declare feature ID, scope, risks, success
  grep -q "### Feature ID" "${spec_dir}/feature-spec.md" || { ok=false; messages+=("Spec missing Feature ID (Article II)"); }
  grep -q "### Risks" "${spec_dir}/feature-spec.md"      || { ok=false; messages+=("Spec missing Risks (Article II)"); }

  # Article XV: DoD gates in plan/tasks
  grep -q "Definition of Done" "${spec_dir}/plan.md"      || { ok=false; messages+=("Plan missing DoD (Article XV)"); }

  # Article XVI: Golden Implementation Patterns
  grep -q "Golden Implementation Patterns" "${spec_dir}/plan.md" || { ok=false; messages+=("Plan missing Golden Patterns (Article XVI)"); }

  # Article XVII-XX: Contract, Tests, Observability, Security
  grep -q "Contract Validation" "${spec_dir}/tasks.md"    || { ok=false; messages+=("Tasks missing contract validation (Article XVIII)"); }
  grep -q "Observability Hooks" "${spec_dir}/tasks.md"    || { ok=false; messages+=("Tasks missing observability (Article XV)"); }
  grep -q "Security Checks" "${spec_dir}/tasks.md"        || { ok=false; messages+=("Tasks missing security (Article XVII)"); }

  # Emit JSON
  printf '{'
  if [ "$ok" = "true" ]; then
    printf '"ok":true,'
  else
    printf '"ok":false,'
  fi
  printf '"messages":['
  for i in "${!messages[@]}"; do
    [[ $i -gt 0 ]] && printf ','
    printf '"%s"' "${messages[$i]}"
  done
  printf ']}\n'
}