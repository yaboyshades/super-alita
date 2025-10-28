# Real-World Task Playbook

This playbook helps operators design and evaluate "real world" tasks for the REUG runtime without compromising reliability or observability guarantees.

## 1. Preparation Checklist

1. **Refresh dependencies**
   - `make deps` (or `uv pip install -r requirements.txt -c constraints.txt`).
2. **Verify baseline health**
   - `pre-commit run --all-files`
   - `pytest -q tests/runtime`
3. **Load environment toggles**
   - Copy `.env.example` → `.env` when needed.
   - Confirm runtime knobs: `REUG_MAX_TOOL_CALLS`, `REUG_EXEC_TIMEOUT_S`, `REUG_SCHEMA_ENFORCE`.
4. **Start telemetry capture**
   - Ensure `telemetry.jsonl` (or configured sink) is writable before trials.

## 2. Framing Real-World Tasks

A useful task couples a concrete business outcome with measurable signals.

| Aspect            | Questions to Ask                                                                 | Example                                                                 |
|-------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Outcome           | What user-visible deliverable confirms success?                                  | Draft a compliance-ready changelog for a security patch.               |
| Inputs            | Which files, prompts, or data sources are in scope?                              | `src/reug_runtime/router.py`, latest guardrail policies.               |
| Constraints       | What safety rails must remain untouched?                                         | Streaming contract `<tool_call>` → `<tool_result>` → `<final_answer>`. |
| Observability     | What telemetry events must appear?                                               | `STATE_TRANSITION`, `TaskStarted`, `TaskSucceeded/Failed`.             |
| Exit Criteria     | How will we know the task is complete?                                           | Target diff merged + regression suite green.                           |

Translate the answers into a concise operator brief. Store the brief in `memory/` or `docs/tasks/` if you want repeatability.

## 3. Routing the Task Through REUG

1. **Select the orchestration entry point**
   - FastAPI: `make run` → call `/v1/chat/stream` or higher-level endpoints (`/sdd/specify`, etc.).
   - CLI: `python -m start_super_alita` for scripted evaluations.
2. **Specify the system instructions**
   - Choose the relevant router context (see `docs/orchestration/rules/*.md`).
   - Override per-run memory by passing `context.overrides` in the API payload when needed.
3. **Seed conversation state**
   - Provide the real-world brief as the initial user message.
   - Add attachments (specs, logs) via `_artifact` references when large.

## 4. Monitoring Execution

Track the closed-loop cognitive model during runs:

1. Confirm `STATE_TRANSITION` events follow the legal path `AWAITING_INPUT → … → RESPONDING_SUCCESS/TaskFailed`.
2. Watch `AbilityCalled`/`AbilitySucceeded` pairs for each span ID.
3. Inspect `ArtifactCreated` events to ensure large outputs are captured without breaking caps.
4. When schema bypass occurs, record justification and follow-up actions.

For live debugging, tail telemetry:

```bash
python -m scripts.tail_telemetry telemetry.jsonl
```

## 5. Evaluation Criteria

| Category            | Signals to Review                                                                                           |
|---------------------|--------------------------------------------------------------------------------------------------------------|
| Reliability         | Retries remain ≤ `REUG_EXEC_MAX_RETRIES`; no circuit breaker trips for core abilities.                       |
| Quality             | Generated diffs respect scope guardrails and coding standards (see `AGENTS.md` and nested policies).         |
| Testing             | Runtime task triggers focused tests; regression suites stay green.                                           |
| Observability       | Telemetry contains hashes (`user_msg_hash`, `args_hash`, `output_hash`) for reproducibility.                 |
| Follow-through      | TODO scoring and bandit updates appear in event stream; tasks conclude with `TaskSucceeded` or `TaskFailed`. |

Document findings after each run in `audit/` or the knowledge graph to inform future bandit weights.

## 6. Post-Run Actions

1. **Archive artifacts**
   - Commit `ArtifactCreated` payloads or summaries to `artifacts/` (git-ignored) for offline review.
2. **Regression reinforcement**
   - Convert new failure modes into tests under `tests/runtime/` or targeted suites.
3. **Prompt/policy tuning**
   - Adjust router instructions when behavior changes can be expressed without code.
4. **Rollback readiness**
   - Capture the command log (commands + exit codes) so failures can be replayed.

## 7. Quick Reference Commands

```bash
# Launch FastAPI runtime
make run

# Fire a real-world brief at the streaming endpoint
http POST :8080/v1/chat/stream messages:='[{"role":"user","content":"<your brief>"}]'

# Run focused runtime tests
pytest -q tests/runtime

# Execute full reliability gate
pre-commit run --all-files && pytest -q tests/runtime
```

Use this playbook as a template: clone it per scenario, fill in the brief, and attach telemetry excerpts so the learning loop stays tight.
