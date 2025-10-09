# REUG Runtime Event Flow

This reference captures the events emitted by `execute_turn` in `src/reug_runtime/loop.py` and how they are surfaced through the Server-Sent Event (SSE) stream in `src/reug_runtime/streaming.py`.

## Event inventory (`execute_turn`)

| Event type | Purpose | Key payload fields |
| --- | --- | --- |
| `MessageOptimized` | Telemetry emitted when the optional message optimizer rewrites the user input prior to orchestration. | `correlation_id`, `len_in`, `len_out`, `steps` |
| `TaskStarted` | Marks the beginning of a turn and records the session correlation metadata. | `correlation_id`, `goal`, `session_id` |
| `KnowledgeContextRetrieved` | Announces that the knowledge graph provided context for the turn, including the related goal identifier. | `correlation_id`, `session_id`, `snippet`, `goal_id` |
| `LLMChunk` | Streams incremental model output tokens (text only). | `data.text` |
| `AbilityCalled` | Indicates that an ability/tool invocation has started and exposes the captured arguments. | `tool`, `correlation_id`, `span_id`, `args` |
| `AbilitySucceeded` | Signals successful tool execution together with the structured result payload. | `tool`, `correlation_id`, `span_id`, `result` |
| `AbilityFailed` | Reports tool execution failure and propagates the error string. | `tool`, `correlation_id`, `span_id`, `error` |
| `KnowledgeAtomCreated` | Confirms persistence of the final answer into the knowledge graph. | `correlation_id`, `session_id`, `atom_id`, `atom_type` |
| `LoopAlignmentTelemetry` | Encodes the closed-loop telemetry snapshot for bandit/reward processing at the end of the turn. | `correlation_id`, `session_id`, `atoms`, `bonds`, `energy`, `todo`, `bandit`, `reward` |
| `KnowledgeBondCreated` | Emits each bond that links the resolved goal to the generated answer atom. | `correlation_id`, `session_id`, `bond_type`, `source_atom_id`, `target_atom_id` |
| `TaskSucceeded` | Final event that carries the assistant reply payload for downstream consumers. | `correlation_id`, `session_id`, `data` |

## SSE event name mapping (`sse_transformer`)

| Internal event type | SSE alias | Notes |
| --- | --- | --- |
| `TaskStarted` | `start` | Beginning of turn lifecycle. |
| `LLMChunk` | `content` | Payload reshaped to `{ "content": text }`. |
| `AbilityCalled` | `tool_start` | Signals tool invocation start. |
| `AbilitySucceeded` | `tool_result` | Streams tool results. |
| `AbilityFailed` | `tool_error` | Streams tool errors. |
| `TaskSucceeded` | `done` | Indicates final response completion. |
| *Any other type* | `message` | Default alias used for telemetry-only events (e.g., knowledge graph or alignment signals). |
