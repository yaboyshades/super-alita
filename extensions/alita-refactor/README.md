# Alita Refactor (VS Code Extension)

Autonomous Refactoring Agent for your workspace, wired to `tools/refactor_hotspots.py`.

## Features

- Scan current workspace or a selected folder for refactor hotspots
- Produces a JSON report `refactor_report_vscode.json` in your workspace
- Registers a (proposed) Copilot Chat participant `@refactor` (Insiders + proposed API)

## Commands

- `Alita: Scan Workspace for Refactor Hotspots`
- `Alita: Scan Folder for Refactor Hotspots`

## Requirements

- Python environment that can run `tools/refactor_hotspots.py`
- Optional: Mangle gRPC + stubs to enable semantic-only mode

## Dev

```
cd extensions/alita-refactor
npm install
npm run compile
```

Then launch the extension host from VS Code.

