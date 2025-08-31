# Telemetry

Tool wrappers in `mcp_server_wrapper.py` emit events to a JSON Lines log
(`telemetry.jsonl`). The file location can be overridden with the
`SUPER_ALITA_TELEMETRY_FILE` environment variable.

## Event sequence

Each tool invocation records the following events:

1. **AbilityCalled** – emitted before execution begins.
2. **AbilitySucceeded** or **AbilityFailed** – emitted after the tool
   returns or raises an error, including duration and output hash.
3. **ArtifactCreated** – emitted when a tool's result exceeds ~200 KB. The
   full output is stored as an external artifact and the event includes its
   size and SHA-256 hash.

These events help verify tool behaviour during tests and can be inspected by
reading the `telemetry.jsonl` file.
