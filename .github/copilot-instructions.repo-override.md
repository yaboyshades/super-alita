# Super Alita – Repository‑Scoped Copilot Interaction & Output Conventions

These custom instructions refine and extend the global GitHub Copilot (@copilot) guidance specifically for the `yaboyshades/super-alita` repository (Super Alita v3.0: Production-Grade Autonomous Cognitive Agent with Dynamic Capabilities & Neural Architecture).

They ensure:
- Deterministic, parseable responses for automation.
- Architecture‑aware assistance (Strategic / Tactical / Operational layers).
- Safe handling of adaptive assets (prompts, strategies, bandit priors, memory distillation outputs).
- Correct file / artifact formatting (critical for PR reviews, docs, and CI robots).

---

## 1. Core Output Contract

When proposing or modifying files:
- ALWAYS wrap each file's full contents in a code block using:  
  ```<language> name=path/to/file```
- For Markdown files you MUST use FOUR backticks to open and close (```` … ````) so that nested code blocks remain intact.
- Include only one file per code block.
- Provide complete file contents (idempotent replacement), not diffs or fragments.

Examples:

```python name=src/scheduler/executor.py
# full file content here
```

````markdown name=docs/ARCHITECTURE.md
# Title
Some text...

```python
# nested code stays intact
```
````

If the user asks for multiple files, emit multiple consecutive file blocks—no prose between them unless explicitly requested.

---

## 2. Listing Issues or Pull Requests

When the user explicitly requests a list of Issues or PRs:

- Use a single code block with language `list`.
- Use `type="issue"` or `type="pr"` in the block header.
- Do NOT mix issues and PRs in the same list.
- Provide ALL requested items—do not truncate.
- Structure each entry as YAML objects under `data:` with required fields:

```list type="issue"
data:
- url: "https://github.com/owner/repo/issues/123"
  state: "open"
  draft: false
  title: "Improve tool orchestration"
  number: 123
  created_at: "2025-08-20T12:34:56Z"
  closed_at: ""
  merged_at: ""
  labels:
  - "enhancement"
  author: "yaboyshades"
  comments: 4
  assignees_avatar_urls:
  - "https://avatars.githubusercontent.com/u/123456?v=4"
```

Same structure for PRs with `type="pr"`. If a date field is not applicable, leave it as an empty string.

No extraneous commentary outside the block unless the user explicitly asks for analysis.

---

## 3. Architectural Awareness

Use the repository's tri‑layer mental model:

| Layer       | Purpose | Representative Components |
|-------------|---------|---------------------------|
| Strategic   | High-level planning, strategy arms (bandits), policy selection | Strategy selectors, bandit priors, plan generators |
| Tactical    | Workflow graph build, node scheduling, context assembly | Orchestrators, tool selectors, memory retrieval policies |
| Operational | Direct execution of tools, LLM calls, validation nodes | Tool adapters, runtime server endpoints, safety filters |

When explaining or generating code:
- Explicitly identify which layer a proposed module belongs to.
- Preserve separation of concerns: strategic modules should not directly invoke tools; they publish decisions consumed by tactical orchestrators.

---

## 4. Adaptive Assets Policy

Sensitive adaptive artifacts:
- `config/strategies.json`
- `memory/distilled/**`
- `prompts/**`
- `policies/**`
- `tool_manifest/**`

Rules:
1. Never add raw user data or PII to memory distillation outputs.
2. When modifying `strategies.json`, include justification (reward basis / prior shift) if asked.
3. If generating new prompt templates, annotate with:
   - Purpose
   - Input contract (placeholders)
   - Failure modes mitigated
4. For tool manifests, ensure fields: `name`, `version`, `description`, `capabilities`, `cost`, `endpoints`, `auth` (if needed).

---

## 5. Safety & Validation Guidance

Before proposing a feature interacting with external APIs/tools:
- Include fallback / retry rationale.
- Note where safety filters apply (pre-LM prompt assembly, post-generation normalization).
- Suggest metric hooks (latency_ms, tool_failure_rate, reward_proxy).

If the user asks "how," provide:
- Input contract
- Error modes
- Observability points
- Memory update rules (what gets stored; summary vs raw)

---

## 6. Testing & Evaluation Conventions

When generating test code:
- Prefer `pytest` style.
- Name files `test_<scope>.py`.
- For evaluation harness additions, emit deterministic seeding if randomness used.
- Provide synthetic test cases under `evaluation/test_cases/*.json` using:
  ```json
  { "id": "case_id", "input": "User instruction or query", "expected_shape": "short/long/json/tools" }
  ```

---

## 7. Bandit / Strategy Updates

When asked to adjust bandit logic:
- Use explicit algorithm label (e.g., UCB1, ThompsonSamplingBeta, EpsilonGreedy).
- Show updated arm metadata fields: `base_weight`, `algorithm_params`, optional `posterior`.
- Keep recompute scripts pure (no side effects except output file).
- If recommending a change, explain reward signal mapping: quality vs latency vs cost weights.

---

## 8. Memory Distillation Additions

When proposing distillation logic:
- Outline chunking rule (token or semantic boundary).
- Provide summarization heuristic (frequency counts, concept graph merges).
- Include hash or version tag to avoid duplicate commits (`sha256` prefix recommended).

---

## 9. File / Module Generation Rules

When asked to "add" or "refactor":
- Provide entire new file(s) using proper file block format.
- If refactoring an existing file and its content is unknown, either:
  1. Ask the user to supply current content, OR
  2. Use a placeholder clearly marked: `# TODO: integrate with existing logic`.

Never invent unseen large codebases; keep changes minimal and composable.

---

## 10. Output Minimalism & Determinism

Unless the user explicitly requests elaboration:
- Avoid extraneous prose outside file blocks.
- Use stable ordering (e.g., JSON objects pretty-printed with consistent key order if user asks for config).

---

## 11. Error Handling Patterns (Preferred)

For Python async modules:
```python
try:
    result = await tool_client.execute(payload, timeout=3)
except asyncio.TimeoutError:
    logger.warning("Tool timeout: %s", tool_id)
    return FallbackResult(reason="timeout")
except ToolExecutionError as e:
    logger.error("Tool failure %s: %s", tool_id, e)
    return FallbackResult(reason="tool_error")
```

Encourage structured return envelopes:
```python
{"status":"ok","data":{...},"metrics":{"latency_ms":123},"warnings":[]}
```

---

## 12. Security & Secrets

Never hardcode secrets. If user requests secret usage examples:
- Show environment variable loading pattern or dependency injection.
- Mention GitHub Actions secrets mapping if relevant (`${{ secrets.MY_SECRET }}`).

---

## 13. When to Ask for Clarification

Ask the user before proceeding if:
- They request modification of an unknown existing file without supplying it.
- Architectural change conflicts with layering principles.
- They ask for a list (issues/PRs) but scope (repo vs org) is ambiguous.

---

## 14. Tooling / Dependencies

Prefer:
- Lint: `ruff`
- Format: `black`
- Types: `mypy (ignore-missing-imports tolerated initially)`
- Tests: `pytest`
- Packaging / runtime staging lines kept minimal; no global implicit side-effects in `__init__`.

---

## 15. Examples (Good vs Bad)

Good:
````markdown name=docs/NEW_PROMPT_TEMPLATE.md
# Tool Selection Prompt (v1)
Purpose: Encourage agent to pick minimal cost tool set for diagnostic queries.

Inputs:
- {user_query}
- {retrieved_facts}
- {available_tools_json}

Constraints:
- Must enumerate chosen tools in JSON under "tool_plan".

Failure Modes Mitigated:
- Over-selection, missing required diagnostic tool.

```json
{
  "instruction": "Analyze the query and propose the tool_plan JSON ONLY."
}
```
````

Bad:
- Fragmented snippet without file wrapper.
- Mixing issue list with PRs.
- Hardcoding API keys.
- Inventing repository paths that do not exist.

---

## 16. Scope Boundaries

Out of scope unless user explicitly requests:
- Live network calls.
- Real secret values.
- Non-deterministic random seeds in harness examples.

---

## 17. Quick Reference Cheat Sheet

| Task | Action |
|------|--------|
| Add file | Provide full file block |
| Update config JSON | Full file JSON block (pretty) |
| List issues/PRs | `list` code block format |
| Generate doc | 4-backtick markdown file block |
| Refactor unknown file | Ask or stub with TODO |

---

Adhere strictly to these conventions to maintain automated tooling compatibility and repository hygiene.

If uncertain, ask concise clarification before generating large artifacts.

# End of Repository-Specific Copilot Instructions