
# Getting Started

<cite>
**Referenced Files in This Document**   
- [README.md](file://README.md)
- [backend/mcp_server.py](file://backend/mcp_server.py)
- [backend/context_server.py](file://backend/context_server.py)
- [backend/agent_orchestrator.py](file://backend/agent_orchestrator.py)
- [config/services.yaml](file://config/services.yaml)
- [requirements.txt](file://requirements.txt)
- [requirements-gpu.txt](file://requirements-gpu.txt)
- [docker-compose.yml](file://docker-compose.yml)
- [docker/docker-compose.yml](file://docker/docker-compose.yml)
- [docker/docker-compose.redis.yml](file://docker/docker-compose.redis.yml)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites and Environment Requirements](#prerequisites-and-environment-requirements)
3. [Installation Process](#installation-process)
4. [Configuration and Initialization](#configuration-and-initialization)
5. [Starting Core Services](#starting-core-services)
6. [Running Your First Agent Workflow](#running-your-first-agent-workflow)
7. [Copilot Integration](#copilot-integration)
8. [Common Setup Issues and Solutions](#common-setup-issues-and-solutions)
9. [Startup Modes: Development vs Production](#startup-modes-development-vs-production)
10. [Verification and Health Checks](#verification-and-health-checks)

## Introduction
This guide provides a comprehensive walkthrough for setting up the Super Alita framework, an advanced, event-driven AI agent system with modular plugins, MCP integration, knowledge graph, streaming orchestration, and adaptive LLM routing. The framework supports multiple deployment modes and integrates with various external services such as Redis, Ollama, and cloud-based LLM providers. This document will guide you through the installation, configuration, initialization, and verification of the system, ensuring a smooth onboarding experience.

## Prerequisites and Environment Requirements
Before installing Super Alita, ensure your environment meets the following prerequisites:

- **Python Version**: Python 3.9 or higher is required. Verify your Python version using `python --version`.
- **Redis**: Redis is used for optional backend services such as rate limiting and event bus. Install Redis from [redis.io](https://redis.io/download) or use Docker.
- **Ollama**: Ollama is required for local LLM inference. Install Ollama from [ollama.com](https://ollama.com/download) and ensure the service is running.
- **GPU Support (Optional)**: For GPU acceleration, ensure CUDA 12.1 is installed and compatible with your hardware. Install PyTorch with CUDA support using the official index.
- **Environment Variables**: Key environment variables include `LLM_MODEL`, `OLLAMA_HOST`, `REDIS_URL`, and provider-specific keys (e.g., `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`).

**Section sources**
- [README.md](file://README.md#L1-L565)
- [config/services.yaml](file://config/services.yaml#L1-L26)
- [requirements.txt](file://requirements.txt#L1-L61)
- [requirements-gpu.txt](file://requirements-gpu.txt#L1-L9)

## Installation Process
To install Super Alita, follow these steps:

1. **Clone the Repository**: Clone the Super Alita repository from its source.
2. **Set Up Virtual Environment**: Create a virtual environment using `python -m venv .venv` and activate it.
3. **Install Dependencies**: Use the provided Makefile or manually install dependencies:
   - For CPU-only setup: `make deps` or `pip install -e .`
   - For GPU support: Install PyTorch with CUDA and `pip install -r requirements-gpu.txt`
4. **Environment Configuration**: Copy `.env.example` to `.env` and configure necessary variables such as API keys and model paths.

**Section sources**
- [README.md](file://README.md#L1-L565)
- [requirements.txt](file://requirements.txt#L1-L61)
- [requirements-gpu.txt](file://requirements-gpu.txt#L1-L9)

## Configuration and Initialization
Configuration is managed through environment variables and YAML files. Key configuration files include:

- **services.yaml**: Defines LLM providers and external services like Redis and Puter.
- **.env**: Contains sensitive information such as API keys and model endpoints.

Initialize the system by setting up the environment file and ensuring all services are reachable. For example, configure `OLLAMA_HOST` to point to your Ollama instance and set `LLM_MODEL=auto` for dynamic provider selection.

**Section sources**
- [config/services.yaml](file://config/services.yaml#L1-L26)
- [README.md](file://README.md#L1-L565)

## Starting Core Services
Super Alita consists of several core services that can be started independently or via Docker:

1. **MCP Server**: Exposes search and fetch tools via a FastMCP instance. Start with `python backend/mcp_server.py`.
2. **Context Server**: Provides lightweight context indexing and search using ChromaDB and Sentence Transformers. Start with `uvicorn backend.context_server:app --reload --port 5001`.
3. **Agent Orchestrator**: Manages multi-agent execution via an OpenAI-compatible endpoint. Start with `uvicorn backend.agent_orchestrator:app --reload --port 5002`.

Alternatively, use Docker Compose to start all services:
```bash
docker-compose -f docker/docker-compose.yml up
```

**Section sources**
- [backend/mcp_server.py](file://backend/mcp_server.py#L1-L59)
- [backend/context_server.py](file://backend/context_server.py#L1-L127)
- [backend/agent_orchestrator.py](file://backend/agent_orchestrator.py#L1-L135)
- [docker/docker-compose.yml](file://docker/docker-compose.yml#L1-L33)

## Running Your First Agent Workflow
To execute a simple agent workflow:

1. **Start the Development Server**: Run `make run` or `python -m uvicorn src.main:app --reload --port 8080`.
2. **Invoke an Agent**: Use the VS Code command `Alita: Invoke Agent (Ollama)` to send a prompt to the local Ollama model.
3. **Observe Output**: The response will stream to a new Markdown document, demonstrating the agent's capability.

For a more complex workflow, use the agent orchestrator to execute a multi-agent task:
```bash
curl -X POST http://127.0.0.1:5002/swarm/execute -H "Content-Type: application/json" -d '{"prompt": "Refactor the authentication module", "context": {}}'
```

**Section sources**
- [README.md](file://README.md#L1-L565)
- [backend/agent_orchestrator.py](file://backend/agent_orchestrator.py#L1-L135)

## Copilot Integration
Super Alita integrates with GitHub Copilot through the `alita-language-tools` extension:

1. **Install the Extension**: In VS Code Insiders, install the `alita-language-tools` extension from the `extensions/` directory.
2. **Configure Runtime**: Set environment variables `alita.runtime.host` and `alita.ollama.model` in VS Code settings.
3. **Use Commands**: Access commands like `Alita: Chat via Runtime (Stream)` to interact with the local runtime.

Ensure the backend is running and accessible at the configured host and port.

**Section sources**
