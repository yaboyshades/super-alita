# Tool Streaming Interface

Tools registered through `ToolRegistry` can optionally support streaming
output. A tool exposes two complementary execution paths:

- **Non-streaming** via `invoke` which returns the final result.
- **Streaming** via `invoke_stream` (alias: `astream`) which yields chunks as
  they are produced.

## Implementing a Streaming Tool

A tool may implement an `astream` method returning an async generator that
yields string chunks. The callable form should still return the final result so
both paths produce equivalent outputs.

```python
class EchoTool:
    async def __call__(self, text: str) -> str:
        return text.upper()

    async def astream(self, text: str):
        for ch in text.upper():
            yield ch
```

`ToolRegistry.invoke_stream` detects `astream` and forwards each yielded chunk.
If no streaming method is present, the registry falls back to the non-streaming
path and yields the final result as a single chunk. Generators returned directly
from the callable are also supported.

Both `invoke` and `invoke_stream` consume generator outputs to ensure the
aggregated result matches the non-streaming path.
