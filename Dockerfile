# The web UI, for Cloud Run. Nothing in the image is a credential: ClickHouse Cloud and
# Gemini are reached with the environment Cloud Run injects at deploy time (see README).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY data/mx_docs.tsv ./data/mx_docs.tsv
RUN uv sync --frozen --no-dev
# `mcp-clickhouse` is launched by name as a subprocess, so the venv has to be on PATH.
ENV PATH="/app/.venv/bin:$PATH" PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "subtext serve --host 0.0.0.0 --port ${PORT}"]
