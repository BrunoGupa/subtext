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

corpus: ## Build the Mexican corpus + embeddings from a loaded aligned_lines (~1.5 min)
	uv run subtext build-mx
	uv run subtext embed-mx

serve: ## Run the web UI on http://127.0.0.1:8000
	uv run subtext serve

ask: ## Ask the agent a question: make ask Q="..."
	uv run subtext ask "$(Q)"

localise: ## Localise one English line: make localise L="Get in the car."
	uv run subtext localise "$(L)"

demo: env up install schema ## Cold start. Needs the corpus - see `make fetch` first.
	@$(MAKE) --no-print-directory corpus
	@echo ""
	@echo "Ready. Try:"
	@echo "  make localise L='What the fuck?'"
	@echo "  make serve      # web UI on http://127.0.0.1:8000"

test: ## Run the test suite
	uv run pytest -q

.PHONY: help env up down nuke install schema load serve ask eval sweep demo test build-mx build-mx-fast embed-mx precedent deploy

build-mx: ## Build the Mexican corpus from the loaded parallel corpus (~75 s)
	uv run subtext build-mx

build-mx-fast: ## Same, but use the shipped document boundaries instead of deriving them
	uv run subtext build-mx --from-index data/mx_docs.tsv

embed-mx: ## Embed the English side of the Mexican corpus + HNSW index (~2 min, no API cost)
	uv run subtext embed-mx

precedent: ## Find Mexican precedent for a line: make precedent Q="Hurry up!"
	uv run subtext precedent "$(Q)"

deploy: ## Deploy the web UI to Cloud Run. Needs GCP_PROJECT and the .env values in the environment.
	gcloud run deploy subtext --source . --project $(GCP_PROJECT) --region us-east1 \
	  --allow-unauthenticated --max-instances 1 --memory 1Gi --cpu 1 --timeout 120 \
	  --set-env-vars CLICKHOUSE_HOST=$(CLICKHOUSE_HOST),CLICKHOUSE_HTTP_PORT=$(CLICKHOUSE_HTTP_PORT),CLICKHOUSE_SECURE=true,CLICKHOUSE_USER=$(CLICKHOUSE_USER),CLICKHOUSE_DATABASE=$(CLICKHOUSE_DATABASE),SUBTEXT_MODEL=gemini-3.8-flash,EMBEDDING_MODEL=gemini-embedding-001,EMBEDDING_DIM=768,GOOGLE_CLOUD_PROJECT=$(GCP_PROJECT) \
	  --set-secrets CLICKHOUSE_PASSWORD=clickhouse-password:latest,GOOGLE_API_KEY=gemini-key:latest
