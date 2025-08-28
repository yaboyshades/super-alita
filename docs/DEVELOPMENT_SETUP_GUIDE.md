# 🛠️ Super Alita Development Setup Guide

This guide provides comprehensive instructions for setting up a development environment for Super Alita.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or 3.12 (recommended)
- **Node.js**: 18+ (for VS Code extensions)
- **Git**: Latest version
- **Docker**: Optional, for containerized development

### Platform-Specific Notes

#### Windows
```bash
# Install Python from python.org or use Windows Store
# Install Git from git-scm.com
# Consider using Windows Subsystem for Linux (WSL) for better compatibility
```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.12 git node

# Using pyenv for Python version management
brew install pyenv
pyenv install 3.12
pyenv global 3.12
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip git nodejs npm
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yaboyshades/super-alita.git
cd super-alita
```

### 2. Set Up Environment
```bash
# Create environment file
cp .env.example .env

# Edit .env with your configuration
# At minimum, set one LLM provider:
# - OPENAI_API_KEY=your-key-here
# - GEMINI_API_KEY=your-key-here
# - Or configure local Ollama: OLLAMA_HOST=http://127.0.0.1:11434
```

### 3. Install Dependencies

#### Using Make (Recommended)
```bash
make deps  # Installs all dependencies
make env   # Creates .env if it doesn't exist
```

#### Manual Installation
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-test.txt
```

### 4. Validate Installation
```bash
# Test the deployment
python validate_deployment.py

# Run smoke tests
make test-smoke

# Start the development server
make run
# Or manually: uvicorn app:app --reload --port 8080
```

## 🔧 Development Workflow

### Running the Application

#### FastAPI Development Server
```bash
make run
# Serves on http://localhost:8080
# Auto-reloads on code changes
```

#### Core Runtime
```bash
python -m src.main
# Runs the core Super Alita runtime
```

#### Docker Development
```bash
# Build and run with Docker
docker build -t super-alita .
docker run -p 8080:8080 super-alita

# Or use Docker Compose
docker-compose up --build
```

### Testing

#### Run All Tests
```bash
make test
# Or manually: PYTHONPATH=./src pytest -v tests/runtime/
```

#### Quick Smoke Test
```bash
make test-smoke
# Or manually: PYTHONPATH=./src pytest -q tests/runtime/test_router_smoke.py
```

#### Specific Test Categories
```bash
# Integration tests with Redis
pytest -m integration_redis

# Specific test files
pytest tests/core/test_event_bus.py

# With coverage
pytest --cov=src tests/
```

### Code Quality

#### Linting and Formatting
```bash
# Run all pre-commit hooks
make lint
# Or manually: pre-commit run --all-files

# Individual tools
black .                    # Format code
ruff check .              # Lint code
mypy src/                 # Type checking
```

#### Pre-commit Hooks (Recommended)
```bash
pip install pre-commit
pre-commit install
# Now hooks run automatically on commit
```

## 🏗️ Architecture Overview

### Project Structure
```
super-alita/
├── src/                    # Core source code
│   ├── main.py            # Application entry point
│   ├── core/              # Core event system
│   ├── reug_runtime/      # Streaming orchestration
│   ├── plugins/           # Plugin system
│   └── ...
├── app.py                 # FastAPI application
├── tests/                 # Test suites
├── docs/                  # Documentation
├── .github/workflows/     # CI/CD automation
├── cortex/                # Automation and tools
└── extensions/            # VS Code extensions
```

### Key Components
- **Event Bus**: Redis-backed event system (`src/core/event_bus.py`)
- **Router**: Streaming orchestration (`src/reug_runtime/router.py`)
- **Plugins**: Modular plugin architecture (`src/plugins/`)
- **MCP Integration**: Model Context Protocol support
- **Knowledge Graph**: Cognitive fabric with atoms/bonds

## 🔌 VS Code Integration

### Extensions Development
```bash
cd extensions/alita-language-tools
npm ci                     # Install dependencies
npm run compile           # Build extension
```

### Recommended VS Code Settings
Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=./src

# Or use the provided configuration
source .env
```

#### Dependency Conflicts
```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt -r requirements-test.txt
```

#### Redis Connection Issues
```bash
# Install and start Redis locally
# Ubuntu/Debian:
sudo apt install redis-server
sudo systemctl start redis

# macOS:
brew install redis
brew services start redis

# Or use Docker:
docker run -d -p 6379:6379 redis:alpine
```

#### Test Failures
```bash
# Some tests may have syntax errors - this is known
# Focus on smoke tests for core functionality
make test-smoke

# Skip problematic tests
pytest -k "not problematic_test_name"
```

### Performance Tips

#### Dependency Installation
- First-time setup takes ~5 minutes due to ML dependencies
- Use `--no-cache-dir` if disk space is limited
- Consider using `uv` for faster installs: `uv pip install -r requirements.txt`

#### Development Server
- Use `--reload` for auto-restart on changes
- Set `LOG_LEVEL=DEBUG` for detailed logging
- Increase timeouts for slow operations: `REUG_MODEL_STREAM_TIMEOUT_S=120`

## 📚 Additional Resources

- [API Documentation](docs/api/) - Auto-generated API docs
- [Architecture Guide](docs/architecture.md) - System architecture
- [Plugin Development](docs/plugins.md) - Creating custom plugins
- [MCP Integration](docs/mcp_integration.md) - Model Context Protocol
- [Automation Guide](docs/POWERFUL_GITHUB_WORKFLOWS.md) - CI/CD workflows

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes and test**: `make test && make lint`
4. **Commit changes**: `git commit -m 'feat: add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

**Note**: This guide is automatically updated by the development environment workflow.
For issues or improvements, please create an issue with the `documentation` label.
