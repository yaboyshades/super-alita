# Copilot Prompt Optimizer (VS Code Extension)

Optimizes and amplifies your prompts before sending them to a chat model. When VS Code's LM Chat API is available (e.g., with GitHub Copilot installed), it routes the enhanced prompt to a chat model and previews the response. Otherwise, it copies the enhanced prompt so you can paste it into Copilot Chat manually.

## Features

- Optimize: normalize phrasing and whitespace.
- Amplify: inject lightweight repository/file context and a concise role framing.
- Send via LM: use VS Code's language model chat when available.
- Clipboard/Replace: copy to clipboard or replace selection with the enhanced prompt.

## Commands

- `Copilot: Open Optimized Chat` (`copilotPromptOptimizer.startChat`)
- `Copilot: Optimize Selection` (`copilotPromptOptimizer.optimizeSelection`)

## Notes & Limitations

- VS Code extension isolation prevents directly intercepting Copilot Chat messages. This extension provides a wrapper UX and LM routing instead.
- LM routing uses a best‑effort call to VS Code's `lm` API. If unavailable, you'll be prompted to open Copilot Chat and paste the enhanced prompt.

## Development

```bash
cd extensions/copilot-prompt-optimizer
npm install
npm run watch
# Press F5 in VS Code to launch Extension Development Host
```

