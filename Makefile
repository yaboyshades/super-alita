.PHONY: run test test-smoke lint deps env clean help ollama-smoke ollama-run mcp-export mcp-abstract mcp-abstract-consolidated

ifneq (,$(wildcard ./.env))
include .env
export $(shell sed 's/=.*//' .env)
endif

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

deps: ## Install runtime + test deps
	pip install -r requirements.txt -r requirements-test.txt

run: ## Start FastAPI dev server on :8080
        uvicorn app:app --reload --port 8080

test: ## Run full tests (runtime suite)
        PYTHONPATH=./src pytest -v tests/runtime/

test-smoke: ## Quick smoke test
	PYTHONPATH=./src pytest -q tests/runtime/test_router_smoke.py

lint: ## Run pre-commit hooks
        pre-commit run --all-files

# --- Ollama helpers ---
OLLAMA_MODEL ?= llama3.1:8b

ollama-smoke: ## Run a direct Ollama smoke test against OLLAMA_HOST
	python scripts/ollama_smoke.py --model $(OLLAMA_MODEL)

ollama-run: ## Pull model and run Ollama smoke test (requires `ollama` CLI)
	@echo "[ollama-run] pulling $(OLLAMA_MODEL)" && ollama pull $(OLLAMA_MODEL)
	$(MAKE) ollama-smoke

run-ollama: ## Start server with GPT-OSS via Ollama (one-shot setup)
	@echo "[run-ollama] Starting server with LLM_MODEL=ollama:gpt-oss:20b and OLLAMA_HOST=http://127.0.0.1:11434"
	LLM_MODEL=ollama:gpt-oss:20b OLLAMA_HOST=http://127.0.0.1:11434 python -m src.main

run-mcp-backend: ## Start the backend MCP server
        python backend/mcp_server.py

run-skillset-backend: ## Start the backend skillset server
        uvicorn backend.skillset_server:app --reload --port 8001

env: ## Create .env from template
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env"; else echo ".env already exists"; fi

clean: ## Clean caches and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage coverage.xml htmlcov build dist *.egg-info
	rm -rf logs/*

.PHONY: autogen-any
autogen-any: ## Run autogen for specified capability (use DESC="description")
	python scripts/run_autogen.py --desc "$(DESC)" --repo .

mcp-export: ## Export Mangle tools to MCP-Box and rebuild catalog
	python examples/mangle_export_to_mcp.py

mcp-abstract: ## Rebuild MCP-Box index and catalog
	python -c "from src.reug_runtime.mcp_abstractor import abstract_mcp_box; abstract_mcp_box('.mcp_box')"

mcp-abstract-consolidated: ## Rebuild MCP-Box with consolidated catalog
	python -c "from src.reug_runtime.mcp_abstractor import abstract_mcp_box; abstract_mcp_box('.mcp_box', consolidate_catalog=True)"
