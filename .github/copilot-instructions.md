# Super Alita - GitHub Copilot Agent Instructions

**ALWAYS follow these instructions first and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.**

## Overview
Super Alita is an advanced, event-driven AI agent system with modular plugins, MCP (Model Context Protocol) integration, knowledge graph, streaming orchestration, and adaptive LLM routing. The system provides both a Python FastAPI backend and a GitHub Copilot agent extension for seamless integration.

## Working Effectively

### Bootstrap Environment and Dependencies
Always run these commands in order for a fresh setup:

1. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Install dependencies (takes ~5 minutes - NEVER CANCEL):**
   ```bash
   pip install -r requirements.txt -r requirements-test.txt
   # Alternative: make deps
   ```
   - NEVER CANCEL: This takes approximately 5 minutes to complete. Set timeout to 10+ minutes.
   - The installation includes PyTorch (CPU), ML libraries, and all required dependencies.

### Run the Application

**Main Python Backend:**
```bash
# Method 1: Using Make (recommended)
make run

# Method 2: Direct uvicorn command
PYTHONPATH=./src uvicorn src.main:app --host 0.0.0.0 --port 8080

# Method 3: Direct Python module
PYTHONPATH=./src python -m src.main --host 0.0.0.0 --port 8080
```

**Health Check:**
```bash
curl http://127.0.0.1:8080/healthz
# Should return: {"status":"healthy","components":{...}}

curl http://127.0.0.1:8080/health  
# Should return: {"status":"healthy","service":"super-alita"}
```

**GitHub Copilot Agent Extension:**
```bash
cd extensions/copilot-agent
npm install  # Takes ~15 seconds
npm run build  # Takes ~2 seconds  
npm start  # Starts on port 8787
```

Health check for Copilot Agent:
```bash
curl http://localhost:8787/healthz
# Should return: ok
```

### Testing

**Quick Smoke Test:**
```bash
make test-smoke
# Takes ~2 seconds but WILL FAIL in current development state
```

**Full Test Suite:**
```bash
make test
# Alternative: PYTHONPATH=./src pytest -q tests/runtime/
# Takes ~10 seconds, WILL HAVE FAILURES in current development state
```

**IMPORTANT:** The test suite has many failing tests. This is EXPECTED in the current development state. Do not attempt to fix unrelated test failures. Only fix tests directly related to your changes.

### Code Quality and Linting

**Individual Linting (RECOMMENDED):**
```bash
ruff check src/ --no-fix  # Check for issues
ruff check src/ --fix     # Auto-fix issues
black src/                # Format code
```

**Pre-commit (may fail due to network timeouts):**
```bash
pre-commit install
make lint  # May timeout - use individual tools above instead
```

## API Usage and Testing

### Main Chat API
Test the core streaming chat functionality:
```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "test the echo tool", "session_id": "demo"}'
```
Expected response: Streaming text with `<tool_call>`, `<tool_result>`, and `<final_answer>` blocks.

### Copilot Agent API
Test copilot-specific commands:
```bash
curl -s -N -X POST http://localhost:8787/copilot \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"health"}]}'
```
Expected response: Server-Sent Events (SSE) format with health status.

Available Copilot commands:
- `health` - System health check
- `status` - Detailed system metrics  
- `help` - Command reference
- `kg <query>` - Knowledge graph queries
- `decide <policy>` - Optimization decisions

## Architecture and Key Components

### Project Structure
- **`src/`** - Main Python source code
  - `src/main.py` - FastAPI application entry point
  - `src/core/` - Core event bus, plugins, neural atom systems
  - `src/reug_runtime/` - Deployable runtime agent
  - `src/plugins/` - Modular plugin system
  - `src/mcp_local/` - MCP integration
- **`app.py`** - Simple FastAPI wrapper for development
- **`extensions/copilot-agent/`** - GitHub Copilot agent (Node.js/TypeScript)
- **`mcp_server/`** - MCP server implementation
- **`tests/`** - Test suite (mirrors src/ structure)

### Key Technologies
- **Backend:** Python 3.11+, FastAPI, uvicorn
- **Event System:** Redis-based event bus with file fallback  
- **AI/ML:** OpenAI, Google Gemini, sentence-transformers, torch
- **Frontend/Agent:** Node.js, TypeScript, GitHub Copilot SDK
- **Development:** pytest, ruff, black, mypy, pre-commit

## Development Guidelines

### Environment Variables  
Key variables in `.env`:
```
REUG_MAX_TOOL_CALLS=5
REUG_EXEC_TIMEOUT_S=20.0
LLM_MODEL=auto  # Enables automatic provider fallback
GEMINI_API_KEY=your_key_here  # Optional
OPENAI_API_KEY=your_key_here  # Optional
PYTHONPATH=./src
```

### Code Style
- **Python:** Use ruff for linting, black for formatting (88 char line length)
- **TypeScript:** Use ESLint and Prettier in copilot-agent directory
- **Type hints:** Required for all Python code
- **Testing:** Add tests in `tests/` that mirror `src/` structure

### Making Changes
1. **Always test your changes:** Run the application and verify functionality
2. **Lint your code:** Use `ruff check --fix` and `black` before committing
3. **Test relevant functionality:** Start the servers and test affected endpoints
4. **Do not fix unrelated test failures:** Focus only on your changes

## Validation Scenarios

After making changes, ALWAYS test these scenarios:

1. **Basic Health Check:**
   ```bash
   # Start backend
   make run &
   sleep 5
   curl http://127.0.0.1:8080/healthz
   ```

2. **Core API Functionality:**
   ```bash
   curl -X POST http://127.0.0.1:8080/v1/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"message": "hello", "session_id": "test"}'
   ```

3. **Copilot Agent (if modified):**
   ```bash
   cd extensions/copilot-agent
   npm run build && npm start &
   sleep 3
   curl -X POST http://localhost:8787/copilot \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"health"}]}'
   ```

## Common Commands Reference

**Environment Setup:**
```bash
cp .env.example .env
pip install -r requirements.txt -r requirements-test.txt  # 5+ minutes
```

**Development:**
```bash
make run              # Start FastAPI server
make test-smoke       # Quick test (will fail)
make lint             # Run pre-commit (may timeout)
ruff check --fix src/ # Lint and fix Python code
black src/            # Format Python code
```

**Manual Server Commands:**
```bash
# Python backend
PYTHONPATH=./src uvicorn src.main:app --port 8080

# Copilot agent  
cd extensions/copilot-agent && npm start
```

**Health Checks:**
```bash
curl http://127.0.0.1:8080/healthz    # Main API
curl http://localhost:8787/healthz     # Copilot agent
```

## Known Issues and Workarounds

1. **Test Failures:** Many tests fail in current development state - this is expected
2. **Pre-commit Timeouts:** Use individual tools (`ruff`, `black`) instead of `make lint`
3. **MCP Setup Script:** `Setup-MCP.ps1` referenced in docs but not present in repository
4. **Network Dependencies:** Some operations require internet access for package downloads

## CI/CD Pipeline

The repository uses GitHub Actions (`.github/workflows/deps-and-tests.yml`):
- Runs on Python 3.11 and 3.12
- Includes Redis service for testing
- Performs security auditing with pip-audit and safety
- Requires passing lint and type checks
- NEVER CANCEL: CI builds can take 15+ minutes including dependency installation and testing

Always run `ruff check --fix` and `black` locally before pushing to avoid CI failures.
