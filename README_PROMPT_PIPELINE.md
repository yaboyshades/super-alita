# Super Alita Prompt-Optimization Pipeline

This integration connects GitHub Copilot in VS Code with Super Alita's advanced reasoning capabilities to create a powerful prompt-optimization and context-enhancement workflow.

## Features

- **Prompt Optimization**: Templates and scaffolds for better LLM prompting
- **Context Enhancement**: Automatic retrieval of relevant Super Alita context
- **DeepConf Integration**: Uses Super Alita's consensus system for better results
- **VS Code Snippets**: Custom GitHub Copilot snippets for rapid prompt development
- **Interactive Workflow**: VS Code tasks for seamless pipeline execution

## Getting Started

### Prerequisites

- VS Code with GitHub Copilot extension
- Super Alita dependencies installed
- Python 3.8 or higher

### Configuration

1. The VS Code settings are already configured in `.vscode/settings.json`
2. Copilot helper configuration is in `copilot-helpers/copilot.config.js`
3. Prompt templates are defined in `src/templates/templates.json`

### Using the Pipeline

#### Method 1: VS Code Task

1. Press `Ctrl+Shift+P` to open the Command Palette
2. Type "Tasks: Run Task" and select it
3. Choose "Run Prompt Pipeline"
4. Enter your prompt when prompted

#### Method 2: Command Line

```bash
python src/pipeline.py "Your prompt here"
```

### Using Copilot Snippets

In any file, you can use the following snippets:

- `opt-prompt`: Expands to an optimized prompt scaffold
- `ctx-block`: Adds a context block
- `deep-conf`: Adds DeepConf consensus configuration
- `reug`: Adds REUG streaming template

For example, typing `opt-prompt` and pressing Tab will expand to:

```
You are an expert [role] specializing in Super Alita development. "[input]". Provide [format]. Steps:
Constraints: [constraints]
Let's think step by step…
```

## Integration with Super Alita

This pipeline leverages several Super Alita components:

- **DeepConf Consensus**: For high-quality, reliable responses
- **Context Retrieval**: Based on Super Alita's codebase and documentation
- **Event Bus**: For asynchronous processing (when available)
- **REUG Runtime**: For structured response streaming (when available)

## Customization

- Modify `templates.json` to add or change prompt templates
- Update `copilot.config.js` to add new snippets or modify existing ones
- Extend `retriever.py` to include additional context sources

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  User Input     │───>│  Context         │───>│  Enhanced       │
│  via VS Code    │    │  Retrieval       │    │  Prompt         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                     │
┌─────────────────┐    ┌─────────────────┐          │
│  Output         │<───│  LLM            │<─────────┘
│  Formatting     │    │  Invocation     │
└─────────────────┘    └─────────────────┘
```
