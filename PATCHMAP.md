PATCHMAP — Single-Turn REUG Streaming Orchestrator

> A contributor guide to the production-grade streaming endpoint, its invariants, telemetry, and how not to break it.



1) Purpose

This document maps the changes that introduced the single-turn, server-side streaming orchestrator wired into REUG: state transitions, dynamic tool registration, retries, circuit breaking, artifact capping, and KG provenance — with tests that lock the behavior.

2) Scope (what this covers)

/v1/chat/stream FastAPI router implementing a single server-turn tool-loop with client streaming.

REUG observability: state transitions, correlation/span IDs, event hashing, telemetry lifecycles.

Dynamic tools: synthesize contract → health check → register → execute.

Safety & limits: parser buffer cap, schema enforcement / bypass, result capping → artifacts, retries/backoff, circuit breaker, client disconnect handling.

Config surface via env.


3) Files (added/modified)

.pre-commit-config.yaml — Black, Ruff (autofix), whitespace hooks.

pytest.ini — quiet mode, asyncio, integration_redis marker.

reug_runtime/config.py — runtime settings (env-driven).

reug_runtime/router.py — streaming orchestrator.

tests/runtime/ — fakes and tests:

fakes.py — FakeEventBus, FakeKG, FakeAbilityRegistry, FakeLLM.

test_router_smoke.py — happy path.

test_router_dynamic_tool.py — unknown tool → synthesize → register → success.

test_router_retry.py — timeout → retry → success.

test_router_result_cap.py — large outputs → artifact capping.

test_router_circuit_breaker.py — breaker opens after repeated failures.

test_router_disconnect.py — (skipped) placeholder for disconnect sim.



4) Runtime invariants (DO NOT BREAK)

1. Single terminal event per turn
Exactly one of: TaskSucceeded XOR TaskFailed.


2. Per-span exclusivity
For each AbilityCalled(span_id=X), there is exactly one matching AbilitySucceeded(span_id=X) XOR AbilityFailed(span_id=X).


3. Legal REUG path (instrumented at minimum)

AWAITING_INPUT → DECOMPOSE_TASK → SELECT_TOOL
→ (EXECUTE_TOOL → PROCESS_TOOL_RESULT)*
→ RESPONDING_SUCCESS

Optional branch: → CREATING_DYNAMIC_TOOL → EXECUTE_TOOL.


4. Caps & safety

Max tool calls per turn: REUG_MAX_TOOL_CALLS (default 5).

Parser buffer cap: ~2 MB (keeps the trailing tail).

Tool result cap: ~200 KB → creates artifact and injects _artifact handle.

Timeouts respected: model stream and tool execution timeouts.

Retries are bounded: first try + REUG_EXEC_MAX_RETRIES.

Circuit breaker opens after repeated failures for a tool.


5. Observability consistency

Every event carries a correlation_id (per turn).

Every tool call carries a span_id (per attempt lineage).

Hashes (user_msg_hash, args_hash, output_hash) are stable across replays (JSON sorted keys).



5) Streaming protocol (model tags)

The model must produce well-formed, JSON-bearing tags:

Tool call
<tool_call>{"tool":"name","args":{...}}</tool_call>

Tool result (injected by server)
<tool_result tool="name">{...}</tool_result>

Tool error (injected by server)
<tool_error tool="name">{"error":"...","message":"..."}</tool_error>

Final answer
<final_answer>{"content":"...","citations":[...]}</final_answer>


The parser tolerates partial chunks and only acts on complete blocks.

6) Telemetry event vocabulary (minimum)

STATE_TRANSITION — {from, to, correlation_id}

TaskStarted — {correlation_id, goal, user_msg_hash}

AbilityCalled — {correlation_id, span_id, tool, args_hash, attempt, max_attempts}

AbilitySucceeded — {correlation_id, span_id, tool, duration_ms, output_hash}

AbilityFailed — {correlation_id, span_id, tool, duration_ms, error, attempt, max_attempts}

TOOL_REGISTERED — {correlation_id, tool, contract_hash}

SchemaBypass — {tool, args_hash}

ArtifactCreated — {tool, artifact_bytes, sha256}

ToolCircuitOpen — {tool}

TaskSucceeded — {correlation_id, answer_atom_id}

TaskFailed — {correlation_id, reason}


> Rule: Emit STATE_TRANSITION at least for the top-level phases (see §4).



7) Dynamic tool registration (contract-first)

When an unknown tool is requested:

1. Synthesize a minimal contract from the call args (types inferred).


2. Health-check (if supported) → register.


3. Emit TOOL_REGISTERED with contract hash.


4. Resume execution.



If health-check or registration fails, inject <tool_error> and let the model recover (e.g., choose a different tool or fix args).

8) Retries, backoff, and circuit breaking

Timeouts
Tool call: REUG_EXEC_TIMEOUT_S (default 20s).
Model stream: REUG_MODEL_STREAM_TIMEOUT_S (default 60s).

Retries
Attempts: 1 + REUG_EXEC_MAX_RETRIES (default 2 total).
Backoff: exponential from REUG_RETRY_BASE_MS (default 250 ms).

Circuit breaker
After N consecutive failures (threshold 3), ToolCircuitOpen is emitted and further calls short-circuit with <tool_error>{"error":"circuit_open"}.


9) Schema enforcement vs bypass

If REUG_SCHEMA_ENFORCE=true (default): invalid args → <tool_error> injected; model must recover.

If false: execute anyway but emit SchemaBypass (keeps audit trail).


10) Result capping & artifacts

If a tool’s JSON result exceeds the cap:

Store out-of-band (artifact handle with sha256, bytes) and inject:

{"_artifact":{"artifact_id":"<short>","sha256":"<hash>","bytes":N}}

Emit ArtifactCreated.

Leave full object out of LLM context to protect latency and token use.


11) Client disconnects

If the client disconnects mid-stream, emit TaskFailed{reason:"client_disconnected"}, cancel the generator gracefully, and release resources.


12) Configuration (env vars)

VarDefaultMeaning

REUG_MAX_TOOL_CALLS5Upper bound on tool calls per turn
REUG_EXEC_TIMEOUT_S20.0Tool execution timeout (seconds)
REUG_MODEL_STREAM_TIMEOUT_S60.0LLM stream timeout (seconds)
REUG_EXEC_MAX_RETRIES1Additional retries after first attempt
REUG_RETRY_BASE_MS250Exponential backoff base in ms
REUG_SCHEMA_ENFORCEtrueEnforce tool input schemas
REUG_EVENT_LOG_DIRNoneOptional log dir (if EventBus uses it)
REUG_TOOL_REGISTRY_DIRNoneOptional registry dir


13) Test matrix (what guarantees what)

TestGuarantees

test_router_smokeHappy path: tool call → result → final answer; core events present
test_router_dynamic_toolUnknown tool synthesizes/Registers → success events
test_router_retryTimeout → retry → success; matching failure/success telemetry
test_router_result_capBig result → _artifact injection; ArtifactCreated emitted
test_router_circuit_breakerRepeated failures → breaker opens; turn still concludes
test_router_disconnectPlaceholder for client disconnect (skipped in CI)


> Run:
pre-commit run --all-files
pytest -q tests/runtime



14) Extension points

AbilityRegistry: add health_check, register, knows, strict validate_args (already tolerated if absent).

EventBus: switch sinks (NDJSON → Redis/Kafka); keep event fields.

KnowledgeGraph: add richer provenance bonds (USED_TOOL, FAILED_TOOL, RETRIED_TOOL), plus artifact nodes.


15) Observability checks (manual)

One terminal event per correlation_id.

For each span_id, exactly one success/failure.

No SchemaBypass in prod (unless intentionally allowed).

Circuit breaker rate (per tool) below SLO threshold.

Latency histograms for AbilitySucceeded.duration_ms look sane.


16) Rollback plan

Feature-flag the router mount (or API prefix).

Revert to previous handler if invariants break in prod.

Preserve logs; replay correlation IDs to reconstruct turns.



---

Developer quickstart

# Formatting & lint
pre-commit run --all-files

# Tests
pytest -q tests/runtime

Golden rule: If you change the protocol tags, event fields, or limits, update this PATCHMAP and the tests in tests/runtime/ to match.
