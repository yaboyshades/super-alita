# Unified Orchestration Runtime (Initial Draft)

This document defines the first unified orchestration layer for Super Alita.
Goal: provide a single configurable pipeline that deterministically chains existing capabilities (SDD → Planning → Consensus → Code Gen → Validation → Scoring) while emitting a stable event grammar for streaming and tooling.

## Motivation
Current runtime paths are fragmented:
1. `reug_runtime.router.execute_turn` implements a reasoning/acting loop with tool calls.
2. Legacy chat endpoints in `src/main.py` provide fallback streaming with different events.
3. Higher-level pipelines (`ladder_reug_generate`, `hybrid_reasoning_pipeline`) exist only as discrete tools.
4. SDD (specify/plan/tasks) + constitutional validation live in a separate router and are not orchestrated with code generation or validation phases.

The unified orchestrator offers a thin, explicit pipeline with pluggable stages. Each stage is optional and failure-tolerant (soft‑fail with diagnostic metadata). It does **not** replace existing routes yet; instead it provides:

* A new ability: `unified_execute`
* A new streaming endpoint: `GET/POST /v1/unified/stream`
* Consistent event types for UI or downstream consumers.

## Stages
| Order | Key | Description | Tool(s) / Mechanism | Optional |
|-------|-----|-------------|---------------------|----------|
| 1 | `specification` | Generate / validate feature specification (SDD specify) | `POST /sdd/specify` via direct framework call (internal) | Yes |
| 2 | `planning` | Implementation plan derivation | `task_planner` tool or SDD plan | Yes |
| 3 | `tasks` | Task breakdown for execution | SDD tasks or heuristic from plan | Yes |
| 4 | `consensus` | Multi-sample consolidation of intent/spec | `deepconf_consensus`; fallback echo | Yes |
| 5 | `code_generation` | TDD code synthesis + write | `code_synthesize_and_write` (or `code_synthesize`) | Yes |
| 6 | `validation` | Basic validation (pytest + import smoke) | `pytest_run`, `python_import_smoke` | Yes |
| 7 | `scoring` | Reward / shadow evaluation | `shadow_reward_score` (if available) | Yes |
| 8 | `finalize` | Aggregate outputs + summary | Internal | No |

## Event Grammar
All events are emitted through the shared event bus (and mirrored to SSE when streaming):

* `UnifiedRunStarted` { run_id, session_id, prompt, config }
* `UnifiedStageStarted` { run_id, stage }
* `UnifiedStageSucceeded` { run_id, stage, duration_ms, output_summary }
* `UnifiedStageFailed` { run_id, stage, duration_ms, error }
* `UnifiedToolEvent` { run_id, stage, original_event } (pass‑through of `Ability*` events when a stage internally triggers tools)
* `UnifiedRunCompleted` { run_id, success, stages, aggregate }

SSE mapping (initial):

| Internal Event | SSE `event:` | Notes |
|----------------|--------------|-------|
| UnifiedRunStarted | start | includes config |
| UnifiedStageStarted | stage_start | stage name in data |
| UnifiedStageSucceeded | stage_result | success payload |
| UnifiedStageFailed | stage_error | error payload |
| UnifiedToolEvent (Ability*) | passthrough mapped to existing (`tool_start`, `tool_result`, `tool_error`) | reuse existing mapping |
| UnifiedRunCompleted | done | final aggregate |

## Configuration Schema (v0)
```json
{
  "type": "object",
  "properties": {
    "run_id": {"type": "string"},
    "enable_specification": {"type": "boolean", "default": false},
    "enable_planning": {"type": "boolean", "default": true},
    "enable_tasks": {"type": "boolean", "default": false},
    "enable_consensus": {"type": "boolean", "default": true},
    "enable_code_generation": {"type": "boolean", "default": false},
    "enable_validation": {"type": "boolean", "default": false},
    "enable_scoring": {"type": "boolean", "default": false},
    "test_first": {"type": "boolean", "default": true},
    "file_path": {"type": "string"},
    "language": {"type": "string", "default": "python"},
    "timeout_s": {"type": "integer", "default": 120}
  }
}
```

## Error Handling Principles
* Stage failure does **not** abort subsequent enabled stages unless `fatal` (currently only specification parsing errors marked fatal).
* Each failure surfaces `error` and optional `trace` (truncated) in stage output.
* Aggregate result contains `stages` map with `status: success|failed|skipped`.

## Ability Contract: `unified_execute`
Input (subset of config + prompt):
```json
{
  "prompt": "Build a priority queue module",
  "file_path": "src/data/priority_queue.py",
  "enable_code_generation": true,
  "enable_validation": true,
  "enable_scoring": true
}
```
Output:
```json
{
  "run_id": "...",
  "prompt": "...",
  "stages": { "planning": {...}, "consensus": {...}, "code_generation": {...} },
  "aggregate": { "consensus_text": "...", "written_files": ["src/data/priority_queue.py"] }
}
```

## Streaming Endpoint
`POST /v1/unified/stream` (JSON body) OR `GET /v1/unified/stream?q=...`
Returns SSE frames per event grammar above.

## Minimal Guarantees
* Always emits `UnifiedRunStarted` and `UnifiedRunCompleted`.
* Provides fallback consensus (echo of prompt) if `deepconf_consensus` missing or errors.
* Code generation skipped if no `file_path` provided unless explicitly requested.

## Future Enhancements (Not in v0)
* Constitutional compliance gating between stages.
* Dynamic adaptive stage enabling based on intermediate signals.
* Persistent run ledger for audit.
* Multi-turn stateful plan refinement.

---
Status: Draft v0 (implementation accompanies this document).
Last Updated: 2025-09-11
