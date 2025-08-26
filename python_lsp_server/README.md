#
# /python_lsp_server/README.md
#
# Description: Guide for setting up and running the Python-based Language Server.
#

# Alita Python Language Server

This directory contains the Python-based implementation of the Language Server for the Alita language, using the `pygls` library. It can be launched by the main VS Code extension as an alternative to the Node.js server, which is useful for integrating Python-native linting, formatting, or analysis tools.

## Features

- **Diagnostics:** Provides real-time feedback on code quality (e.g., style warnings).
- **Completions:** Suggests keywords and common constructs.

## Setup

The Python dependencies for this server are managed and bundled by the main extension's build process, typically using a tool like `nox` or a custom script.

To set it up for local development:

1.  **Create a Virtual Environment:**
    It's recommended to create a dedicated virtual environment for the server.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**
    Install `pygls` and any other required packages.
    ```bash
    pip install "pygls>=1.3.2"
    ```

## Running

The Language Server is designed to be launched by the VS Code extension client via its `lsp_runner.py` script. The runner communicates over standard I/O (stdio).

To debug it directly from VS Code:
1.  Open the `alita-lang-ext` project.
2.  In `.vscode/launch.json`, there is usually a debug configuration like "Python: Attach" that can connect to a running Python process.
3.  Alternatively, you can configure a "Python: Module" launch configuration to run `lsp_runner` directly.

## Integration

The main VS Code extension's `lspClient.ts` is responsible for configuring and launching this server. It can be modified to point to the `lsp_runner.py` script and pass initialization options.