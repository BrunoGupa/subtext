"""The ADK agent: plan -> retrieve -> generate SQL -> execute -> validate -> speak."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from ..config import settings
from . import clickhouse_mcp
from .tools import ALL_TOOLS, RETRIEVED_KEY, record_line_ids_from_rows, run_sql

APP_NAME = "subtext"

INSTRUCTION = """
You answer questions about a corpus of film and television dialogue stored in ClickHouse.

The questions worth asking are the ones neither SQL nor search can answer alone. There is no
`broke_a_promise` column, so "how many times does this character break a promise?" has to be
resolved by finding the moments semantically and then counting them structurally.

Work in this order, every time:

1. PLAN. Say in one or two sentences what the question needs: is it a lookup (a column holds
   the answer), a meaning question (only vector search can find it), or both? Name the
   character, season or title constraints you spotted.
2. GROUND. Call `inspect_schema` with the user's question before writing any SQL. Use the
   columns it returns and no others — if a column is not in the retrieved schema, do not
   reference it.
3. RETRIEVE.
   - Meaning + counting ("how many times", "how often", "which season most") ->
     `aggregate_semantic_matches`. One query, vector search feeding a GROUP BY.
   - Meaning, no counting ("when does she admit it?") -> `search_dialogue`.
   - Purely structural ("how many lines does Vale have in season 3?") -> `run_query`,
     which executes SQL against ClickHouse. `list_tables` and `list_databases` are there
     if you need to confirm something about the physical schema.
   Combining them is normal and expected.

   One trap worth naming, because it is the most common way to get this wrong: filtering
   by `character` restricts to lines that character SPEAKS. Questions about what a
   character DID are usually evidenced by someone else's line — "you said an hour",
   "you swore" — so use the `involving` argument, which matches speaker OR addressee.
   Filtering a "how many times did X..." question by `character = X` will look correct
   and quietly return almost nothing.
4. RECOVER. If a tool returns no rows, do not answer from memory and do not pad the answer.
   Try once more with the meaning phrased differently, or a wider `max_distance`, or without
   the character filter. If it is still empty, say plainly that the corpus does not support
   an answer, and say what you looked for. An honest empty result is a correct answer.
5. VALIDATE. Before you speak, call `verify_answer` with your draft and every line_id it
   relies on. If it comes back not ok, fix the answer — remove what is unsupported or
   retrieve again. Never present an answer that failed verification.

When you answer:
- Lead with the number or the finding, then the evidence.
- Quote the dialogue you are relying on with its season, episode, character and timecode.
- Say how you got it: which was semantic, which was SQL.
- Distinguish what the data shows from what you are inferring. "Break a promise" is a
  judgement about meaning; the corpus records dialogue, not intentions. Say so when the
  distinction matters, and never round a fuzzy semantic match up into a hard fact.
""".strip()


def _after_tool(tool, args, tool_context, tool_response):  # noqa: ANN001
    """Record line ids returned by the MCP SQL tool.

    The retrieval tools record what they return so `verify_answer` has something to
    check against. The MCP toolset cannot — it is an external process that knows
    nothing about this project's session state — so its rows are harvested here
    instead. Without this, SQL-derived citations would look unsupported.
    """
    if getattr(tool, "name", "") == "run_query":
        record_line_ids_from_rows(tool_context, tool_response)
    return None


def build_agent(model: str | None = None, *, use_mcp: bool = True) -> LlmAgent:
    """The agent.

    `use_mcp=True` (the default, and what the ClickHouse track requires) routes SQL
    through the official `mcp-clickhouse` server. `use_mcp=False` falls back to the
    in-process `run_sql` tool, which is useful when the MCP server cannot start.
    """
    tools: list = list(ALL_TOOLS)
    if use_mcp:
        tools.append(clickhouse_mcp.build_toolset())
    else:
        tools.append(run_sql)

    return LlmAgent(
        name="subtext_agent",
        model=model or settings().gemini_model,
        description="Answers questions about a film dialogue corpus using hybrid retrieval over ClickHouse.",
        instruction=INSTRUCTION,
        tools=tools,
        after_tool_callback=_after_tool,
    )


@dataclass
class AgentAnswer:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_line_ids: list[int] = field(default_factory=list)
    verified: bool | None = None

    @property
    def tools_used(self) -> list[str]:
        return [call["name"] for call in self.tool_calls]


async def ask_async(
    question: str,
    *,
    model: str | None = None,
    user_id: str = "local",
    use_mcp: bool = True,
) -> AgentAnswer:
    runner = InMemoryRunner(agent=build_agent(model, use_mcp=use_mcp), app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=user_id)

    message = types.Content(role="user", parts=[types.Part(text=question)])
    answer = AgentAnswer(text="")

    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=message
    ):
        for call in event.get_function_calls() or []:
            answer.tool_calls.append({"name": call.name, "args": dict(call.args or {})})
        for response in event.get_function_responses() or []:
            if response.name == "verify_answer":
                payload = response.response or {}
                answer.verified = bool(payload.get("ok"))
        if event.is_final_response() and event.content and event.content.parts:
            answer.text = "".join(part.text or "" for part in event.content.parts).strip()

    final = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session.id
    )
    if final is not None:
        answer.retrieved_line_ids = list(final.state.get(RETRIEVED_KEY, []))
    return answer


def ask(question: str, *, model: str | None = None, use_mcp: bool = True) -> AgentAnswer:
    """Synchronous wrapper around :func:`ask_async`."""
    import asyncio

    return asyncio.run(
        ask_async(question, model=model, user_id=f"cli-{uuid.uuid4().hex[:8]}", use_mcp=use_mcp)
    )


# ADK's `adk web` / `adk run` discovery hook.
root_agent = None


def _lazy_root_agent() -> LlmAgent:
    global root_agent
    if root_agent is None:
        root_agent = build_agent()
    return root_agent
