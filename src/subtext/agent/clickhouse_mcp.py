"""The official ClickHouse MCP server, wired in as an ADK toolset.

The hackathon's ClickHouse track requires that the project "actively use ClickHouse at
runtime via the official ClickHouse MCP server (mcp-clickhouse)". So the agent's SQL
path runs through `mcp-clickhouse` over stdio rather than through this project's own
ClickHouse client: when the model writes a `SELECT`, it is the MCP server that executes
it.

The retrieval tools in `tools.py` still use `clickhouse-connect` directly, because they
have to embed the question in Python before they can query at all — an embedding is not
something SQL text can carry. The division is honest and worth being able to state:
**structured queries go through MCP; the vector path goes through the driver.**

`mcp-clickhouse` connects as a read-only user of its own accord, so a `DROP` reaching it
fails at the server with `READONLY` — a second, independent layer under `sql_guard`.
"""

from __future__ import annotations

import os
import shutil

from ..config import settings

SERVER_COMMAND = "mcp-clickhouse"
MCP_QUERY_TOOLS = ("run_query", "list_databases", "list_tables")


def server_env() -> dict[str, str]:
    """Environment for the MCP server subprocess.

    `mcp-clickhouse` reads `CLICKHOUSE_PORT`, while this project distinguishes the HTTP
    and native ports; the translation happens here rather than in `.env`.
    """
    s = settings()
    return {
        "CLICKHOUSE_HOST": s.ch_host,
        "CLICKHOUSE_PORT": str(s.ch_port),
        "CLICKHOUSE_USER": s.ch_user,
        "CLICKHOUSE_PASSWORD": s.ch_password,
        "CLICKHOUSE_SECURE": "true" if s.ch_secure else "false",
        "CLICKHOUSE_DATABASE": s.ch_database,
        # A subprocess with no PATH cannot find its own interpreter.
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }


def is_available() -> bool:
    return shutil.which(SERVER_COMMAND) is not None


def build_toolset():
    """An `McpToolset` speaking to `mcp-clickhouse` over stdio."""
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from mcp import StdioServerParameters

    if not is_available():
        raise RuntimeError(
            f"{SERVER_COMMAND} is not on PATH. Run `uv sync` (it is a project dependency), "
            "or start the agent with use_mcp=False to fall back to the built-in SQL tool."
        )

    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=SERVER_COMMAND,
                args=[],
                env=server_env(),
            ),
            timeout=30.0,
        ),
        tool_filter=list(MCP_QUERY_TOOLS),
    )
