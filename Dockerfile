# The web UI, for Cloud Run. Nothing in the image is a credential: Gemini is reached with the
# environment Cloud Run injects at deploy time (see README). The database is in the image:
# ClickHouse runs embedded (chDB) over the five tables the page reads, built below from
# Parquet exported by `make runtime-data`. No database server to keep up or pay for.
FROM python:3.12-slim AS app
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

# Built in its own stage so the Parquet never ships: only the finished chDB directory does.
FROM app AS data
COPY data/runtime ./data/runtime
RUN SUBTEXT_DB=chdb python -c "from subtext.db import build_chdb; build_chdb('data/runtime', '/app/chdb')"

FROM app
COPY --from=data /app/chdb /app/chdb
ENV SUBTEXT_DB=chdb SUBTEXT_CHDB_PATH=/app/chdb
EXPOSE 8080
CMD ["sh", "-c", "subtext serve --host 0.0.0.0 --port ${PORT}"]
