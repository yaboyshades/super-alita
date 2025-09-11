# Deep Research for Copilot Chat

VS Code Chat participant `@research` that performs web/academic/technical searches (SearXNG or Perplexica) and synthesizes results with your configured Language Model via the VS Code LM API.

## Features

- `@research` chat participant for Copilot Chat
- Modes: `/academic`, `/technical`, or default web
- Backends: SearXNG or Perplexica
- Fetches and extracts page text, summarizes with inline citations `[n]`
- Configurable via Settings → Deep Research
 - Optional DeepCode pipeline: append `/implement[:task_kind]` and `/apply` to run native DeepCode generation and optionally apply results

## Setup

1) Install dependencies in this folder:

```
cd extensions/deep-research
npm install
npm run compile
```

2) Run the extension (press F5 in VS Code) or package via `vsce` if desired.

3) Configure settings:

- `deepResearch.provider`: `searxng` or `perplexica`
- `deepResearch.searxng.endpoint`: e.g., `http://localhost:8080`
- `deepResearch.perplexica.baseUrl`: e.g., `http://localhost:3000`
- (Optional) `deepResearch.model.vendor` / `deepResearch.model.family` to pick a specific LM

## Usage

In Copilot Chat:

- `@research latest developments in vector databases`
- `@research /academic RAG evaluation best practices 2024`
- `@research /technical streaming architectures with WebSockets`
 - `@research RAG eval best practices 2025 /academic /implement:text2backend` (research → synthesize → send synthesis as DeepCode requirements)
 - `@research build a simple web scraper for product pages /implement:web_scraper /apply` (will apply the generated changes)

## Notes

- If no LM is available via `vscode.lm`, results are returned with links and basic aggregation.
- Network errors per-source are ignored; successful fetches are summarized.
- This is a minimal, privacy-friendly baseline. Extend with richer extractors or scholarly APIs as needed.
 - DeepCode pipeline uses `tools/deepcode_cli.py` under your workspace; ensure your Python venv is set up (the extension tries common `.venv` paths or falls back to `python`).
