# REUG Runtime User Input Flow

This document explains how a single user message traverses the Super Alita runtime, from the HTTP request boundary through the orchestration loop, tool execution, telemetry, and streaming response surfaces. It highlights how the subsystems cooperate and where the primary extension points live.

## 1. Application Startup & Shared State

`create_app` in `src/main.py` loads environment variables, configures logging, and wires shared components into the FastAPI application state: an event bus, the ability registry, the in-memory knowledge graph, and an LLM client. It also exposes health probes and optional API-key/rate-limit middleware.【F:src/main.py†L1505-L1628】【F:src/main.py†L486-L676】【F:src/main.py†L720-L842】

Key objects placed on `app.state`:

- `event_bus`: defaults to the JSONL-backed `FileEventBus`, or a richer pub/sub bus when configured.【F:src/reug_runtime/event_bus.py†L18-L87】
- `ability_registry`: a schema-aware registry with built-in tools and support for dynamic registration/execution.【F:src/main.py†L486-L676】
- `kg`: `SimpleKG`, which supplies lightweight context retrieval and captures atoms/bonds generated during a turn.【F:src/main.py†L720-L842】
- `llm_model`: obtained via `reug_runtime.llm_client.get_llm_client`, providing an async `stream_chat` API that the orchestrator consumes.【F:src/main.py†L1505-L1599】

This shared state is consumed by every chat request so the downstream orchestration logic can remain framework-agnostic.

## 2. HTTP Surfaces

Two complementary API layers exist:

1. **Unified Chat REST API** (`src/api/chat_endpoints.py`): exposes `/api/chat/session`, `/api/chat/history/{id}`, `/api/chat/message`, and `/api/chat/stream`. Each request resolves a `UnifiedChatService` instance and defers to its `stream_turn` generator. `/message` buffers the streamed chunks into a single reply, while `/stream` forwards the stream as SSE frames.【F:src/api/chat_endpoints.py†L1-L87】
2. **REUG Streaming Router** (`src/reug_runtime/router.py`): serves `/v1/chat/stream` (POST/GET). It performs rate-limit checks, emits agent-to-agent telemetry, and then streams the orchestration events through the SSE transformer.【F:src/reug_runtime/router.py†L43-L134】

Both surfaces ultimately delegate to the same orchestration generator, keeping the execution semantics uniform.

## 3. Session & Consensus Service

`UnifiedChatService` maintains in-memory sessions, stores user/assistant messages, and bridges the API layer to the runtime loop. When `stream_turn` is invoked it:

1. Adds the user message to the session history.
2. Iterates over `execute_turn` (imported from the runtime router), yielding every event downstream.
3. Collects `LLMChunk` text into a final assistant message and persists it in the session store.
4. Optionally invokes the `deepconf_consensus` tool after the base reply to append a consensus summary and emit a `ConsensusResult` event.【F:src/unified_chat/chat_service.py†L1-L134】

This abstraction keeps HTTP handlers thin while still supporting additional post-processing features like consensus.

## 4. Core Turn Execution (`execute_turn`)

`execute_turn` in `src/reug_runtime/loop.py` is the heart of the runtime:

1. **Guardrails** – applies Maestro security hardening and optional message optimization middleware, emitting telemetry when optimizers run.【F:src/reug_runtime/loop.py†L24-L146】【F:src/reug_runtime/loop.py†L188-L236】
2. **Task Lifecycle Telemetry** – emits `TaskStarted` and state transition events (`STATE_TRANSITION`) to keep the orchestration finite-state machine observable.【F:src/reug_runtime/loop.py†L238-L308】
3. **Context Assembly** – constructs the system prompt, optionally pulls knowledge graph context, and records retrieval telemetry.【F:src/reug_runtime/loop.py†L310-L382】
4. **Reasoning Loop** – instantiates an `Orchestrator` and repeatedly:
   - Streams LLM output chunks, capturing potential tool calls and forwarding each chunk as `LLMChunk` events.【F:src/reug_runtime/loop.py†L384-L451】【F:src/reug_runtime/loop.py†L64-L137】
   - Appends assistant messages (with tool-call metadata) to the conversation state.【F:src/reug_runtime/loop.py†L452-L474】
5. **Acting Loop** – for each tool call:
   - Emits `AbilityCalled` telemetry, ensures the tool is registered (via `ToolCatalogService`), executes it through the ability registry, and emits success/failure events while building tool response messages.【F:src/reug_runtime/loop.py†L476-L566】【F:src/reug_runtime/tools/service.py†L14-L147】
   - Adds `<tool_result>` snippets back into the message list so the model can incorporate results on subsequent reasoning passes.【F:src/reug_runtime/loop.py†L566-L604】
6. **Finalization** – normalizes the final answer, persists it to the knowledge graph (creating atoms/bonds), emits alignment telemetry, and finishes with a `TaskSucceeded` event containing the structured final answer payload.【F:src/reug_runtime/loop.py†L606-L707】

Because `execute_turn` is an async generator, every emitted dict immediately flows to whichever streaming surface invoked it.

## 5. Tooling & Ability Registry

The default `SimpleAbilityRegistry` offers built-in tools (e.g., `echo`, `brainstorm_mcp_stub`, `deepconf_consensus`), JSON Schema validation, and hooks for registering custom executors. During execution the orchestrator routes each tool call here, so downstream integrations only need to implement the registry interface to extend functionality.【F:src/main.py†L486-L676】【F:src/reug_runtime/tools/__init__.py†L1-L86】

`ToolCatalogService` centralizes dynamic registration, persistence (via `.mcp_box`), and heuristics for auto-registering tools discovered in model output. It prevents duplicate registrations and keeps the registry synchronized with persisted MCP specs.【F:src/reug_runtime/tools/service.py†L14-L147】【F:src/reug_runtime/tools/service.py†L149-L249】

## 6. Knowledge Graph Integration

`SimpleKG` is a placeholder knowledge store that illustrates the contract expected by the loop: it can return contextual snippets, map sessions to goals, and persist new atoms/bonds created when the agent produces answers. The loop emits `KnowledgeContextRetrieved`, `KnowledgeAtomCreated`, and `KnowledgeBondCreated` telemetry as these interactions occur, enabling external observers to mirror the state transitions in real time.【F:src/main.py†L720-L842】【F:src/reug_runtime/loop.py†L310-L707】

## 7. Streaming & Telemetry Surfaces

`src/reug_runtime/streaming.py` converts the raw event generator into SSE frames, mapping internal event types to stable event names and injecting optional heartbeat pings. Both FastAPI routers use this transformer, so clients receive consistent SSE envelopes regardless of which endpoint they hit.【F:src/reug_runtime/streaming.py†L1-L64】

The event bus hierarchy allows pluggable observability. The default `FileEventBus` appends JSONL traces, while `InMemoryPubSubEventBus` adds pub/sub semantics and maintains a cached view of agent state mutations. These buses are drop-in replacements because they share the `BaseEventBus` interface consumed by the loop and router telemetry emitters.【F:src/reug_runtime/event_bus.py†L18-L122】

## 8. Putting It Together

1. A client POSTs `/api/chat/stream` (or `/v1/chat/stream`). API middleware enforces keys/rate limits, then the handler resolves shared state from the FastAPI app and calls `UnifiedChatService.stream_turn` with the user message.【F:src/api/chat_endpoints.py†L28-L82】【F:src/reug_runtime/router.py†L78-L134】
2. `UnifiedChatService` records the message and iterates over `execute_turn`, yielding each event. The service also appends the assistant reply (and optional consensus follow-up) to the session log.【F:src/unified_chat/chat_service.py†L39-L134】
3. `execute_turn` orchestrates reasoning, tool execution, knowledge graph updates, and telemetry emission. Every yielded event flows through `sse_transformer`, becoming SSE frames delivered to the client stream.【F:src/reug_runtime/loop.py†L188-L707】【F:src/reug_runtime/streaming.py†L15-L64】
4. The client renders streamed `LLMChunk` content, observes tool activity via telemetry events, and receives the final structured answer, all while the event bus persists the trace for auditing.【F:src/reug_runtime/event_bus.py†L18-L87】

With this pipeline the components operate cohesively: shared state seeded at startup, HTTP routers acting as thin veneers, the orchestrator enforcing guardrails and telemetry, and auxiliary services (registry, knowledge graph, event bus) providing persistence and extensibility points. Running the runtime’s regression suite (`pytest -q tests/runtime`) exercises these integrations to confirm the streaming loop, tool dispatch, and telemetry wiring continue to work together as designed.【F:tests/runtime/test_execute_turn_stream.py†L1-L95】
