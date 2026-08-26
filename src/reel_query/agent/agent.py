"""The ADK agent: plan -> retrieve -> generate SQL -> execute -> validate -> speak."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from ..config import settings
from .tools import ALL_TOOLS, RETRIEVED_KEY

APP_NAME = "reel_query"

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
   - Purely structural ("how many lines does Vale have in season 3?") -> `run_sql`.
   Combining them is normal and expected.
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


def build_agent(model: str | None = None) -> LlmAgent:
    return LlmAgent(
        name="reel_query_agent",
        model=model or settings().gemini_model,
        description="Answers questions about a film dialogue corpus using hybrid retrieval over ClickHouse.",
        instruction=INSTRUCTION,
        tools=list(ALL_TOOLS),
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


async def ask_async(question: str, *, model: str | None = None, user_id: str = "local") -> AgentAnswer:
    runner = InMemoryRunner(agent=build_agent(model), app_name=APP_NAME)
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


def ask(question: str, *, model: str | None = None) -> AgentAnswer:
    """Synchronous wrapper around :func:`ask_async`."""
    import asyncio

    return asyncio.run(ask_async(question, model=model, user_id=f"cli-{uuid.uuid4().hex[:8]}"))


# ADK's `adk web` / `adk run` discovery hook.
root_agent = None


def _lazy_root_agent() -> LlmAgent:
    global root_agent
    if root_agent is None:
        root_agent = build_agent()
    return root_agent
