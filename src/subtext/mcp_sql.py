"""Read the corpus through the official ClickHouse MCP server.

The ClickHouse track requires that the project "actively use ClickHouse at runtime via the
official ClickHouse MCP server (mcp-clickhouse)". The question-answering agent in
`agent/agent.py` has always done that, but it is not the product: a reader who opens the
web UI and types a line never touched it. This module puts the MCP server in the path of
the thing people actually use.

**What goes through here and what does not.** The phrase channel is SQL and nothing else,
so it runs through the server. The vector channel cannot: the cue has to be embedded in
Python before there is a query to send, and shipping a 768-float literal through a text
protocol on every keystroke would be slower and no more honest. That division is the same
one `agent/clickhouse_mcp.py` documents, and it is worth being able to state plainly:
**structured queries go through MCP; the vector path goes through the driver.**

The server is a subprocess speaking stdio, and MCP is asynchronous while the retrieval code
is not. Rather than make every caller async -- or pay to start a process per query -- one
session is opened on first use and kept on a background event loop for the life of the
process.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import json
import re
import shutil
import threading
from typing import Any, Sequence

from .agent.clickhouse_mcp import SERVER_COMMAND, server_env

#: `run_query` is the only tool this module needs. `mcp-clickhouse` runs it read-only.
QUERY_TOOL = "run_query"


class _Bridge:
    """One MCP session, one background loop, callable from synchronous code."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._session: Any = None
        self._stack: Any = None
        self._lock = threading.Lock()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _open(self) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        command = shutil.which(SERVER_COMMAND)
        if command is None:
            raise RuntimeError(
                f"{SERVER_COMMAND} is not on PATH. It is a project dependency -- run `uv sync`."
            )
        self._stack = AsyncExitStack()
        # The server narrates every connection and query on stderr, which the child
        # inherits from us -- twenty lines of INFO over the top of a one-line answer. It
        # is kept, not discarded: `SUBTEXT_MCP_LOG` names a file when the trace is wanted.
        log_path = os.getenv("SUBTEXT_MCP_LOG")
        errlog = open(log_path, "a") if log_path else open(os.devnull, "w")
        self._stack.callback(errlog.close)
        read, write = await self._stack.enter_async_context(
            stdio_client(
                StdioServerParameters(command=command, args=[], env=server_env()),
                errlog=errlog,
            )
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def _call(self, sql: str) -> list[list[Any]]:
        result = await self._session.call_tool(QUERY_TOOL, {"query": sql})
        text = result.content[0].text
        payload = json.loads(text)
        if isinstance(payload, dict) and "rows" in payload:
            return payload["rows"]
        raise RuntimeError(f"unexpected reply from {QUERY_TOOL}: {text[:200]}")

    def query(self, sql: str) -> list[list[Any]]:
        # The lock guards opening the session, not the calls: MCP numbers its requests,
        # so one session answers several at once, and the phrase channel's ten queries
        # used to wait for each other here -- 4 s of a request that spends most of that
        # waiting on the network.
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._submit(self._open())
        return self._submit(self._call(sql))

    def close(self) -> None:
        if self._stack is not None:
            try:
                self._submit(self._stack.aclose())
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)


_bridge: _Bridge | None = None


def bridge() -> _Bridge:
    global _bridge
    if _bridge is None:
        _bridge = _Bridge()
        atexit.register(_bridge.close)
    return _bridge


def quote(value: str) -> str:
    """A ClickHouse string literal.

    `run_query` takes SQL text and nothing else -- the MCP tool has no bound parameters --
    so values are inlined here instead of by the driver. Everything that reaches this
    function is a phrase cut out of the user's own cue by `tokenizer.tokens`, which keeps
    letters and apostrophes and drops the rest, but the escaping does not rely on that:
    backslash first, then the quote, which is the order ClickHouse's own parser expects.
    """
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


_SELECT = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def query(sql: str) -> list[list[Any]]:
    """Run one read-only statement through the MCP server."""
    if not _SELECT.match(sql) or ";" in sql.rstrip().rstrip(";"):
        raise ValueError("only a single SELECT or WITH statement may go through MCP")
    return bridge().query(sql)


def query_values(sql: str) -> list[tuple]:
    """Rows as tuples, so callers read the same shape the driver returns."""
    return [tuple(row) for row in query(sql)]


def available() -> bool:
    return shutil.which(SERVER_COMMAND) is not None
