# REUG Runtime Configuration Toggles

This guide catalogs the runtime environment flags that shape the behavior of
the REUG event loop, tool execution pipeline, and large language model (LLM)
clients. Each toggle lists its backing environment variable, default value,
the modules that consume the setting, and the operational impact on the
runtime.

## Loop and Tool Execution Controls

| `SETTINGS` attribute | Environment variable | Default | Referenced in | Impact |
| --- | --- | --- | --- | --- |
| `max_tool_calls` | `REUG_MAX_TOOL_CALLS` | `5` | `loop.execute_turn` | Caps the number of reasoning/acting iterations per turn to prevent runaway tool loops. |
| `tool_timeout_s` | `REUG_EXEC_TIMEOUT_S` | `20.0` seconds | `loop.Orchestrator._acting_step`, `router_tools.call_tool_async` | Per-invocation timeout for abilities; exceeding the limit yields `AbilityFailed` telemetry and error content. |
| `model_stream_timeout_s` | `REUG_MODEL_STREAM_TIMEOUT_S` | `60.0` seconds | `loop.Orchestrator._reasoning_step`, `llm_client.*`, `router_tools.stream_with_timeout` | Upper bound for awaiting model streaming chunks; protects against stalled providers. |
| `schema_enforce` | `REUG_SCHEMA_ENFORCE` | `True` | `Settings` (consumed by router/tooling layers) | Governs whether tool input payloads must satisfy their declared schema before execution. |

## Message Optimizer Pipeline

| `SETTINGS` attribute | Environment variable | Default | Referenced in | Impact |
| --- | --- | --- | --- | --- |
| `message_optimizer_enabled` | `REUG_MESSAGE_OPTIMIZER_ENABLED` | `True` | `loop.execute_turn` | Enables the message middleware chain that amplifies/optimizes the user prompt before model invocation. |
| `message_optimizer_emit_telemetry` | `REUG_MESSAGE_OPTIMIZER_TELEMETRY` | `True` | `loop.execute_turn` | Emits `MessageOptimized` telemetry events describing optimizer steps and token deltas. |
| `message_optimizer_max_len` | `REUG_MESSAGE_OPTIMIZER_MAX_LEN` | `6000` characters | `loop.execute_turn` | Soft caps the optimized prompt length to avoid unbounded growth prior to streaming. |

## LLM Generation Controls

| `SETTINGS` attribute | Environment variable | Default | Referenced in | Impact |
| --- | --- | --- | --- | --- |
| `default_temperature` | `REUG_DEFAULT_TEMPERATURE` | `0.2` | `llm_client.create_chat_completion` (and related helpers) | Baseline sampling temperature when the upstream provider does not supply one. |
| `max_retries` | `REUG_EXEC_MAX_RETRIES` | `1` | `Settings` | Number of retry attempts after the first failure for execution strategies that opt-in to Settings. |
| `retry_base_ms` | `REUG_RETRY_BASE_MS` | `250` milliseconds | `Settings` | Base backoff interval for retrying execution requests. |
| `copilot_context` | `REUG_COPILOT_CONTEXT` | `True` | `Settings` | Indicates whether REUG should request additional Copilot context when available. |

## Observability and Storage

| `SETTINGS` attribute | Environment variable | Default | Referenced in | Impact |
| --- | --- | --- | --- | --- |
| `event_log_dir` | `REUG_EVENT_LOG_DIR` | `None` | `Settings` | Optional filesystem target for persisting emitted telemetry. |
| `tool_registry_dir` | `REUG_TOOL_REGISTRY_DIR` | `None` | `Settings` | Directory used by the dynamic tool catalog to persist registrations. |
| `api_prefix` | `API_PREFIX` | `/` | `Settings` | Prefix applied to REST routes when mounting the runtime API. |

## Formatting and Contract Enforcement

| Toggle | Environment variable | Default | Referenced in | Impact |
| --- | --- | --- | --- | --- |
| Output contract prompt injection | `ALITA_FORMAT_CONTRACT` | `false` | `loop.execute_turn` | When enabled, appends formatting rules to the system prompt so the model receives contract guidance during reasoning. |
| Final answer normalization | `ALITA_FORMAT_ENFORCE` | `false` | `loop.execute_turn` | Post-processes the final assistant message through `normalize_output_contract` to enforce markdown/code layout guarantees before emitting telemetry and final responses. |

### Working with Boolean Toggles

Boolean flags treat the values `"1"`, `"true"`, `"yes"`, and `"on"` (case
insensitive) as `True`; all other values fall back to `False`. Clearing an
environment variable reverts to the defaults listed above.

### Operational Guidance

- **Timeout hygiene:** Increase `REUG_EXEC_TIMEOUT_S` or
  `REUG_MODEL_STREAM_TIMEOUT_S` for slower tools/models, but keep
  `REUG_MAX_TOOL_CALLS` conservative to avoid long-running loops.
- **Optimizer governance:** Disable `REUG_MESSAGE_OPTIMIZER_ENABLED` or shrink
  `REUG_MESSAGE_OPTIMIZER_MAX_LEN` when upstream prompts must remain untouched
  (e.g., deterministic regression tests).
- **Schema discipline:** Leave `REUG_SCHEMA_ENFORCE` enabled in production so
  malformed tool arguments trigger guardrails instead of reaching adapters.
- **Formatting compliance:** Combine `ALITA_FORMAT_CONTRACT` and
  `ALITA_FORMAT_ENFORCE` to both instruct the model and normalize its final
  output when strict formatting guarantees are required.
