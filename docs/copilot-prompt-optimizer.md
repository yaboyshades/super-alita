# Architecting a Self-Aware Development Loop: Copilot, Custom Agents, and VS Code via MCP

## Overview

This document outlines Super Alita's implementation of a sophisticated AI-assisted development ecosystem that integrates GitHub Copilot, custom agents, and VS Code through the Model Context Protocol (MCP). This creates a self-aware feedback loop where AI tools collaborate to build and improve AI agents.

## Architecture Components

### 1. VS Code Integration Layer

- **Extension Path**: `extensions/copilot-prompt-optimizer`
- **MCP Server**: `src/vscode_integration/mcp_stdio_server.py`
- **Agent Workflow**: `src/vscode_integration/agent_workflow_config.py`

### 2. Enhanced Consensus Integration

- **Consensus Provider**: Leverages Super Alita's `enhanced_consensus_ability.py`
- **Tool Registration**: Integrated with existing ability registry
- **Streaming Support**: Works with REUG streaming architecture

### 3. Repository Self-Wrapping

- **Repo Tool**: `src/vscode_integration/repo_mcp_tool.py`
- **Paper Ingestion**: `src/vscode_integration/paper_ingestion_tool.py`
- **Workflow Orchestration**: Policy-driven agent collaboration

## Why This Approach

The traditional approach of intercepting Copilot Chat messages is limited by VS Code's extension isolation. Our solution provides:

1. **MCP-Based Tool Integration**: Direct integration with Super Alita's tool ecosystem
2. **Policy-Driven Collaboration**: Custom instructions that govern AI behavior
3. **Self-Aware Feedback Loop**: Agents that can analyze and improve their own code
4. **Enhanced Consensus**: Multi-response aggregation for better decision making

## Implementation Guide

### Prerequisites

Follow Super Alita's standard environment setup:

```bash
# Environment setup
cp .env.example .env
pip install -r requirements.txt -r requirements-test.txt  # 5 minutes
uvicorn app:app --reload --port 8080

# Health check
curl http://127.0.0.1:8080/healthz
```

### 1. MCP Server Setup

The MCP stdio server exposes Super Alita's capabilities to VS Code:

```python
# Located at src/vscode_integration/mcp_stdio_server.py
# Exposes tools: repo_read_file, repo_write_file, paper_extract_text,
# deepconf_consensus, repo_search_code, etc.
```

### 2. VS Code Configuration

Create `.vscode/mcp.json` to connect VS Code to the MCP server:

```json
{
  "servers": {
    "super-alita-tools": {
      "type": "stdio",
      "command": "${workspaceFolder}/.venv/Scripts/python.exe",
      "args": ["${workspaceFolder}/src/vscode_integration/mcp_stdio_server.py"],
      "description": "Super Alita development tools and agent capabilities"
    }
  }
}
```

### 3. Custom Instructions

The `.github/copilot-instructions.md` file acts as the "director's script":

- **Tool Prioritization**: Forces Copilot to use MCP tools over internal knowledge
- **Safety Protocols**: Requires approval for file modifications and terminal commands
- **Quality Enforcement**: Mandates Black formatting and Ruff linting
- **Read-Before-Write**: Prevents file corruption through required content inspection

### 4. Extension Development

For the VS Code extension component:

```bash
cd extensions/copilot-prompt-optimizer

# First-time setup (creates package-lock.json)
npm install  # 40 seconds

# Subsequent builds
npm run compile  # 3 seconds

# Development mode
npm run watch  # Continuous compilation
```

**Note**: Use `npm install` for first-time setup to generate `package-lock.json`, then use `npm ci` for subsequent clean installs.

## Workflow Examples

### Research Paper to Code Translation

```bash
# 1. Download and process paper
curl -X POST http://127.0.0.1:8080/ability/execute/paper_download \
  -d '{"url": "https://arxiv.org/pdf/2312.00752.pdf", "filename": "attention_mechanisms.pdf"}'

# 2. Extract algorithmic content
curl -X POST http://127.0.0.1:8080/ability/execute/paper_generate_summary \
  -d '{"pdf_path": "attention_mechanisms.pdf", "focus_areas": ["methods", "algorithms"]}'

# 3. Generate implementation via consensus
curl -X POST http://127.0.0.1:8080/ability/execute/deepconf_consensus \
  -d '{"prompt": "Implement the attention mechanism from the paper summary", "method": "weighted_vote"}'
```

### Self-Improvement Loop

1. **Analysis Phase**: Copilot uses `repo_search_code` to identify improvement opportunities
2. **Planning Phase**: Enhanced consensus generates multiple refactoring approaches
3. **Implementation Phase**: `repo_write_file` applies changes with diff previews
4. **Validation Phase**: Automatic quality checks via Ruff and Black
5. **Iteration**: Self-correction based on test results and linting feedback

## VS Code Extension Usage

### Installing the Extension

1. **Build the Extension**:

   ```bash
   cd extensions/copilot-prompt-optimizer
   npm install
   npm run compile
   ```

2. **Install in VS Code**:
   - Press `F5` to launch Extension Development Host, OR
   - Package and install: `vsce package && code --install-extension copilot-prompt-optimizer-0.1.0.vsix`

### Using the Extension

1. **Open Optimized Chat**: Use Command Palette (`Ctrl+Shift+P`) → "Copilot Optimizer: Open Optimized Chat"
2. **Enter Prompt**: Type your development question or request
3. **Enhanced Response**:
   - If Language Model API is available: Response opens in new document
   - If not available: Optimized prompt copied to clipboard for manual paste

### Configuration

Access settings via `Ctrl+Shift+P` → "Copilot Optimizer: Configure Prompt Optimization":

- `copilotOptimizer.enabled`: Enable/disable optimization
- `copilotOptimizer.mcpServerPath`: Path to Super Alita MCP server
- `copilotOptimizer.includeRepoContext`: Include repository context
- `copilotOptimizer.useEnhancedConsensus`: Use consensus algorithms

## Enhanced Features

### Code Snippet Library

The extension includes a comprehensive snippet library with Super Alita-specific patterns:

#### Available Snippets

- **Standard Python**: File headers, imports, loops, functions, classes
- **Super Alita Plugins**: `alitaplugin` - Creates plugin following `PluginInterface` patterns
- **Enhanced Consensus**: `consensus` - Uses `EnhancedConsensusProvider` with proper config
- **Async Testing**: `asynctest` - Creates async tests with `@pytest.mark.asyncio`
- **MCP Tools**: `mcptool` - Registers tools with ability registry

#### Creating Custom Snippets

1. Select code in editor
2. Run `Copilot Optimizer: Create Snippet from Selection`
3. Enter name, prefix, and description
4. Snippet automatically added to language-specific snippet file

#### Snippet Best Practices

- Follow Black 88-character line length
- Use proper type hints per Pylance strict mode
- Include comprehensive docstrings
- Follow Super Alita naming conventions

### Integration Commands

```bash
# Create snippet from selection
Ctrl+Shift+P → "Copilot Optimizer: Create Snippet from Selection"

# Bridge to AI extensions
Ctrl+Shift+P → "Copilot Optimizer: Bridge Extensions"

# Start optimized chat
Ctrl+Shift+P → "Copilot Optimizer: Start Optimized Chat"
```

## Performance Optimization

### Resource Management

- **Process Monitoring**: Use VS Code's Process Explorer (Help > Open Process Explorer)
- **Memory Limits**: Set `"workbench.extensionHost.maxMemory": 4096` in settings.json
- **File Exclusions**: Configure `files.watcherExclude` for large directories

### Consensus Optimization

- **Method Selection**: Use `weighted_vote` (default) for balanced accuracy/speed
- **Sample Tuning**: Adjust `num_samples` based on complexity (2-5 recommended)
- **Temperature Range**: Use 0.2 range for diverse but focused responses

## Security Best Practices

### Tool Safety

- **Approval Gates**: All file modifications require explicit user approval
- **Sandbox Execution**: Use `src/sandbox/exec_sandbox.py` for dynamic code execution
- **Process Safety**: All terminal commands via `src/core/proc.py` (no `shell=True`)

### Secret Management

- **Environment Variables**: Store secrets in `.env` files (gitignored)
- **VS Code SecretStorage**: Use native keychain integration for sensitive data
- **API Key Rotation**: Regular rotation of Ollama and external service keys

## Troubleshooting

### Common Issues

1. **npm ci Error (EUSAGE)**

   - **Cause**: Missing `package-lock.json` file
   - **Solution**: Run `npm install` first to generate lockfile, then use `npm ci`

2. **Copilot Ignores MCP Tools**

   - Check `.vscode/mcp.json` syntax
   - Verify MCP server process is running
   - Strengthen custom instructions with "MUST" directives

3. **Consensus Tool Not Found**

   - Validate tool registration: `curl http://127.0.0.1:8080/tools/catalog`
   - Check for `deepconf_consensus` in tool list
   - Restart server if tools missing

4. **Extension Activation Failed**
   - Check VS Code Developer Console (`Help > Toggle Developer Tools`)
   - Verify TypeScript compilation: `npm run compile`
   - Ensure Python virtual environment is active

### Validation Commands

```bash
# System health
python validate_deployment.py

# Tool catalog check
curl http://127.0.0.1:8080/tools/catalog | jq '.[] | select(.id=="deepconf_consensus")'

# MCP server test
.venv/Scripts/python.exe src/vscode_integration/mcp_stdio_server.py

# Extension compilation
cd extensions/copilot-prompt-optimizer && npm run compile
```

## Integration with Super Alita

This system seamlessly integrates with Super Alita's existing architecture:

- **Event Bus**: All tool executions emit events via the event bus
- **Ability Registry**: MCP tools are registered through the standard ability interface
- **REUG Streaming**: Compatible with streaming workflow orchestration
- **Enhanced Consensus**: Direct integration with 5 consensus algorithms
- **Plugin System**: Follows `PluginInterface` patterns for consistency

## Future Enhancements

- **Multi-Agent Collaboration**: Orchestrate multiple AI agents through MCP
- **Adaptive Instructions**: Dynamic instruction modification based on task success
- **Code Review Integration**: Automated PR review using consensus algorithms
- **Knowledge Graph Integration**: Leverage Super Alita's knowledge graph for context

## Conclusion

This architecture represents a significant advancement in AI-assisted development, creating a symbiotic relationship between human developers and AI tools. By following Super Alita's established patterns and leveraging MCP for tool integration, we achieve a self-aware development loop that accelerates innovation while maintaining security and quality standards.

The system transforms GitHub Copilot from a passive assistant into an active, policy-driven collaborator that understands the custom agent's context and capabilities, enabling unprecedented levels of AI-assisted development productivity.
