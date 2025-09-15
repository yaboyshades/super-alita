# Bash JSON Array Serialization Patterns

This note captures three small, copy/paste-ready strategies for turning Bash arrays into JSON with correct commas, quoting, and validation checks. Each pattern is backed by the prototype script [`scripts/serialize-json-array-examples.sh`](../scripts/serialize-json-array-examples.sh).

## Sample data set

The examples share a deliberately tricky array so that spaces, quotes, Unicode, and newline characters are all covered:

```bash
sample_values=(
  "alpha"
  "beta gamma"
  "value with \"quotes\""
  $'line with\nnewline'
  'unicode ☃'
)
```

## Pattern 1 — `jq --args`

Use jq’s `$ARGS.positional` helper to convert every argument into a JSON string and pack the result into an array. jq handles the quoting/escaping, so the shell just needs to forward the array elements as distinct arguments.

```bash
build_with_jq() {
  jq -n '$ARGS.positional' --args "$@"
}
```

**When to use**: any script that already depends on `jq`, or whenever you want the most compact and reliable option. This is the recommended default.

**Validation tip**: pipe the output back into `jq -e .` to assert that it stayed valid JSON.

## Pattern 2 — Manual `printf` join with escaping

When `jq` is unavailable (for example, inside a minimized container), you can still build the JSON yourself. The `render_json_string` helper streams characters to stdout with only the control characters we actually care about, and the loop appends each item and comma as it goes.

```bash
render_json_string() {
  local input=${1-}
  local length=${#input}
  local i char

  for (( i = 0; i < length; i++ )); do
    char=${input:i:1}
    case $char in
      '"') printf '\\"' ;;
      '\\') printf '\\\\' ;;
      $'\n') printf '\\n' ;;
      $'\r') printf '\\r' ;;
      $'\t') printf '\\t' ;;
      *) printf '%s' "$char" ;;
    esac
  done
}

build_with_printf() {
  local elements=("$@")
  local first=1

  printf '['
  for element in "${elements[@]}"; do
    if (( first )); then
      first=0
    else
      printf ','
    fi

    printf '"'
    render_json_string "$element"
    printf '"'
  done
  printf ']\n'
}
```

**When to use**: controlled environments where you can document and enforce the character set. Extend `render_json_string` if you expect other control characters.

**Validation tip**: call the `validate_json` helper from the script (or run `jq -e . <<<"$(build_with_printf ...)"`) during tests to guard against regressions.

## Pattern 3 — Here-doc assembly for structured payloads

Many scripts need to embed an array inside a larger JSON document. Compose the array first, then splice the lines into a here-doc with predictable indentation. The `$(if ...; then printf ...)` block emits comma-terminated lines and cleanly handles the empty-array case.

```bash
build_here_doc() {
  local elements=("$@")
  local last_index=$(( ${#elements[@]} - 1 ))
  local lines=()

  for i in "${!elements[@]}"; do
    local escaped suffix
    escaped=$(render_json_string "${elements[$i]}")
    suffix=",";
    if (( i == last_index )); then
      suffix=""
    fi
    lines+=("    \"${escaped}\"${suffix}")
  done

  cat <<JSON
{
  "items": [
$(if ((${#lines[@]} > 0)); then printf '%s\n' "${lines[@]}"; fi)
  ]
}
JSON
}
```

**When to use**: scripts that generate request bodies or configuration files. The pattern keeps indentation readable while guaranteeing that commas only appear between elements.

**Validation tip**: capture the here-doc via command substitution and feed it to `jq -e .` just like the standalone array.

## Reuse checklist

1. Decide whether you can depend on `jq`. Prefer Pattern 1 if possible.
2. If you must stay POSIX-only, vendor the `render_json_string` helper and keep unit tests around it.
3. Always run `jq -e .` (or another JSON validator) in CI to catch quoting mistakes early.
4. Store reusable helpers (`render_json_string`, `build_here_doc`) in a shared library script if multiple automation tasks need them.
5. Document edge cases (null bytes, binary blobs) before extending the patterns — those require more sophisticated tooling.

## Prototype script

Run the example script to see all three strategies side-by-side along with validation output:

```bash
./scripts/serialize-json-array-examples.sh
```

It emits each serialized form and confirms JSON correctness with jq’s evaluator. Use this script as a quick regression harness when evolving future tooling.
