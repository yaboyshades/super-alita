.PHONY: run test test-smoke lint deps env clean help

ifneq (,$(wildcard ./.env))
include .env
export $(shell sed 's/=.*//' .env)
endif

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

deps: ## Install runtime + test deps
	pip install -q -r requirements.txt -r requirements-test.txt

run: ## Start FastAPI dev server on :8080
	uvicorn app:app --reload --port 8080

test: ## Run full tests (runtime suite)
	PYTHONPATH=./src pytest -v tests/runtime/

test-smoke: ## Quick smoke test
	PYTHONPATH=./src pytest -q tests/runtime/test_router_smoke.py

lint: ## Run pre-commit hooks
	pre-commit run --all-files

env: ## Create .env from template
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env"; else echo ".env already exists"; fi

clean: ## Remove caches and logs
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage coverage.xml htmlcov build dist *.egg-info
	rm -rf logs/*

deepcode-test:
	pytest -q tests/plugins/test_deepcode_orchestrator.py tests/test_deepcode_integration.py
