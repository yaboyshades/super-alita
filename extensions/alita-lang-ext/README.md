# Alita Language Extension

Provides basic language support for the Alita language, including:

- Syntax highlighting via TextMate grammar.
- Snippets for common patterns.
- Language Server Protocol features through a bundled server.
- Semantic tokens, task provider, and debug adapter wiring.

## Ollama Agent (Local LLM)

This extension can invoke a local Ollama model via the command palette:

- Run `Alita: Invoke Agent (Ollama)` and enter a prompt.
- Configure settings `alita.ollama.host` (default `http://127.0.0.1:11434`) and `alita.ollama.model` (e.g., `llama3.1:8b`). If no model is set, you’ll be prompted on first use.

Requires an Ollama server running locally. See https://ollama.com for installation and models.

## Telemetry

The extension records anonymous activation events to help improve the
extension. See `telemetry.json` for details and disable collection via
VS Code settings if desired.
