# Orchestrator Acting Step ↔ Tool Catalog Service Contract

This document captures the runtime contract between the orchestration loop and
`ToolCatalogService.ensure_tool_registered`. The contract formalizes how
`Orchestrator._acting_step` prepares tool execution and how the catalog service
ensures tools exist before execution begins.

## Call Site

`Orchestrator._acting_step` iterates through LLM supplied tool calls. For each
normalized tool call it performs the following sequence:

1. Extract the tool identifier (`tool_name`) and deserialize arguments into a
   plain `dict[str, Any]` (`tool_args`).
2. Emit `AbilityCalled` telemetry on the event bus and yield the same event to
   streaming consumers.
3. Invoke `ToolCatalogService.ensure_tool_registered(tool_name, tool_args, registry)`.
4. Await `registry.execute(tool_name, tool_args)` to run the ability. Success or
   failure emits the corresponding telemetry (`AbilitySucceeded` or
   `AbilityFailed`) using the same `span_id` issued in step 2.

## Arguments Passed to the Catalog Service

The call to `ensure_tool_registered` uses the following values:

- `tool_name`: string identifier chosen by the LLM (falls back to empty string
  if unavailable, causing the helper to no-op).
- `tool_args`: sanitized dictionary derived from the tool call payload. Any
  deserialization errors collapse to an empty mapping to preserve safety.
- `registry`: the orchestrator's ability registry instance. It exposes the
  runtime API used for introspection (`knows`) and execution (`execute`).

These values allow the catalog service to evaluate heuristics for dynamic tool
registration without additional context.

## Expected Side Effects

`ensure_tool_registered` is expected to:

- Return `True` when the tool is available for immediate execution. This may be
  because the registry already knows the tool or because the service registered
  it dynamically.
- Register missing tools opportunistically by calling
  `registry.register_tool(...)` using heuristics implemented in
  `_auto_register_tool`. When triggered, the service also persists an MCP tool
  specification under the configured MCP box directory via
  `register_dynamic_tool`.
- Leave the registry untouched (and return `False`) only when registration fails
  or throws an exception. `_acting_step` currently proceeds regardless but
  relies on the registry to raise if execution is impossible.
- Avoid emitting telemetry directly; `_acting_step` retains responsibility for
  Ability event emission so span correlation remains intact.

## Ordering Guarantees

`_acting_step` always invokes `ensure_tool_registered` **before** awaiting
`registry.execute`. This ordering guarantees that any dynamic registration logic
completes (or fails fast) before the registry is asked to execute the tool. The
caller continues emitting `AbilitySucceeded`/`AbilityFailed` telemetry using the
same span so downstream consumers observe a consistent Ability lifecycle.

