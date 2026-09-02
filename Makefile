.DEFAULT_GOAL := help
SHELL := /bin/bash

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example with a generated ClickHouse password
	@test -f .env && { echo ".env already exists - leaving it alone"; exit 0; } || true
	@sed "s|^CLICKHOUSE_PASSWORD=.*|CLICKHOUSE_PASSWORD=$$(openssl rand -hex 24)|" .env.example > .env
	@echo "wrote .env (gitignored). Add your GOOGLE_API_KEY to it."

up: ## Start ClickHouse
	docker compose up -d --wait

down: ## Stop ClickHouse (keeps data)
	docker compose down

nuke: ## Stop ClickHouse and delete its data volume
	docker compose down -v

install: ## Install python dependencies
	uv sync

schema: ## Create tables
	uv run subtext init-db

fetch: ## Download the third-party corpora (~4.2 GB, not redistributed here) - see CORPUS.md
	uv run subtext fetch

load: ## Load the bundled sample corpus and embed it
	uv run subtext load --source sample
	uv run subtext embed
	uv run subtext embed-schema

serve: ## Run the web UI on http://127.0.0.1:8000
	uv run subtext serve

ask: ## Ask the agent a question: make ask Q="..."
	uv run subtext ask "$(Q)"

eval: ## Run the golden-set evaluation
	uv run subtext eval

sweep: ## Sweep chunk strategies x top-k
	uv run subtext sweep

demo: env up install schema load ## Full cold start to a queryable corpus
	@echo "Ready. Run 'make serve' for the web UI, or:"
	@echo "  make ask Q='How many times does Vale break a promise?'"

test: ## Run the test suite
	uv run pytest -q

.PHONY: help env up down nuke install schema load serve ask eval sweep demo test
