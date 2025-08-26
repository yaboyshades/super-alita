# Copilot Prompt Engineering

## Context mapping

Adapters gather local signals and convert them into the structured payload that
`build_copilot_context()` consumes. The table below documents the current
mapping:

| Adapter signal | `build_copilot_context` field | Notes |
| --- | --- | --- |
| User message | `prompt` | Raw user request after redaction. |
| File snippets (`path`, `excerpt`) | `files` | Each entry hashed for dedupe and capped to budget. |
| Lint or build diagnostics | `diagnostics` | `{path, line, message}` tuples. |
| Target tests | `tests` | Links to unit tests that exercise the code. |

## Safety and token budgets

* **Token budgets** – keep the assembled context below the model threshold.
  Gemini adapters compress oversized contexts when the estimated tokens exceed
  8K tokens【F:src/core/gemini_pilot.py†L154-L170】.
* **Scrubbing policy** – sensitive strings (emails, tokens, secrets) are masked
  before leaving the adapter via `redact_prompt_and_context()`【F:src/core/utils/redaction.py†L1-L54】.
* **Deterministic hash usage** – adapters hash each context segment using
  `sha256_json()` so telemetry and caching remain stable across retries【F:src/core/utils/hash_utils.py†L25-L28】.

## Usage example

```python
from src.core.utils import redact_prompt_and_context, sha256_json
from src.core.gemini_pilot import build_copilot_context

prompt = "Refactor app.py to use FastAPI"  # raw user input
files = [{"path": "app.py", "excerpt": "def main():..."}]
diagnostics = []
tests = ["tests/runtime/test_router_smoke.py"]

# 1. scrub
prompt, context, _ = redact_prompt_and_context(prompt, {"files": files})
# 2. hash
context_hash = sha256_json(context)
# 3. build final payload
copilot_ctx = build_copilot_context(prompt=prompt, files=files,
                                   diagnostics=diagnostics, tests=tests,
                                   hash=context_hash)
```

## Tests

The following tests exercise the redaction and adapter pathways:

- [tests/core_utils/test_utils_core.py](../tests/core_utils/test_utils_core.py)
- [tests/plugins/test_cortex_adapter.py](../tests/plugins/test_cortex_adapter.py)

## Feeding `build_copilot_context()`

Adapters should gather minimal, scrubbed snippets, respect token budgets, and
compute deterministic hashes before calling `build_copilot_context()`. The
resulting payload is then streamed to the runtime, which inserts it into the LLM
conversation.
