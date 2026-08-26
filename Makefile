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
	uv run reel-query init-db

load: ## Load the bundled sample corpus and embed it
	uv run reel-query load --source sample
	uv run reel-query embed
	uv run reel-query embed-schema

ask: ## Ask the agent a question: make ask Q="..."
	uv run reel-query ask "$(Q)"

eval: ## Run the golden-set evaluation
	uv run reel-query eval

sweep: ## Sweep chunk strategies x top-k
	uv run reel-query sweep

demo: env up install schema load ## Full cold start to a queryable corpus
	@echo "Ready. Try: make ask Q='How many times does Vale break a promise?'"

test: ## Run the test suite
	uv run pytest -q

.PHONY: help env up down nuke install schema load ask eval sweep demo test
