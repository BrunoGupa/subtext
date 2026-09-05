"""The localisation agent: three steps, one of them a model.

    EvidenceAgent   (BaseAgent) -- retrieve precedent          no tokens
    translator      (LlmAgent)  -- write the Spanish           ONE call
    RegisterGate    (BaseAgent) -- accept, or send back once   no tokens

Wrapped in a `LoopAgent` with `max_iterations=2`, so a cue costs one Gemini call, or two
if the first attempt used peninsular Spanish. **The loop is closed by Python, not by the
model**: the gate counts words against the same lexicons that built the corpus and sets
`escalate` when the line is clean. Nothing asks Gemini whether it is happy with its own
output, and nothing lets it decide whether to retrieve.

That division follows the project's own rule (see the `deterministic-vs-llm-in-adk`
skill): a step that makes no creative decision should not be an LlmAgent. Retrieval
tokenizes and looks things up; the gate counts words. Only the translation is judgement.

Contrast the question-answering agent in `agent.py`, which *does* loop over tools —
correctly, because there the choice of tool is the work. Here every cue takes the same
path, so a tool-calling loop would buy nothing but latency and a chance to skip the
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.genai import types

from ..config import settings
from .agent import APP_NAME
from ..localise import (
    INSTRUCTION,
    RETRY_SUFFIX,
    check_register,
    format_evidence,
    gather_evidence,
)

#: Reasoning is off deliberately, and not only to save money. With thinking on the model
#: reached past the evidence -- "Right now." became *"Ahorita mismo."*, which it invented,
#: instead of *"Ahorita."*, which the corpus attests 1,444 times. Grounding is the product;
#: letting the model deliberate its way off the evidence defeats it.
GENERATION_CONFIG = types.GenerateContentConfig(
    # Back to 0.2 at Bruno's call, and the argument for 0 does not survive scrutiny: its
    # only justification was reproducibility, and temperature 0 does NOT deliver that here.
    # Repeated calls still disagree across processes because reasoning runs despite
    # thinking_budget=0 and takes a different path each time. Since determinism is not on
    # offer either way, translation quality decides, and a little sampling reads better
    # than a flat argmax.
    temperature=0.2,
    # 2048, not the 256 that looks generous for a one-line answer. Measured 2026-09-04 on
    # gemini-3.8-flash with the real 1,018-token prompt:
    #
    #     max_output_tokens=256   -> thoughts=205..245, and MAX_TOKENS truncation
    #     max_output_tokens=2048  -> thoughts=None, clean STOP, 12 output tokens
    #
    # A tight cap does not merely clip the answer: with one set, the model reasons anyway
    # despite thinking_budget=0, and those 240-odd thinking tokens are charged against the
    # same cap, leaving ~11 for the reply. That is what cut "Le voy a hacer una oferta que"
    # mid-sentence in the first run. The cap costs nothing when unused -- billing is on
    # tokens produced (12), not on the ceiling.
    max_output_tokens=2048,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)


def _event(ctx: InvocationContext, author: str, text: str,
           actions: EventActions | None = None) -> Event:
    return Event(
        invocation_id=ctx.invocation_id,
        author=author,
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=actions or EventActions(),
    )


class EvidenceAgent(BaseAgent):
    """Retrieve phrase precedent and neighbours for the cue. No model, no decisions."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        actions = EventActions()
        state = ctx.session.state
        cue = state.get("cue", "")

        if state.get("evidence_block"):
            # Second pass of the loop: the evidence has not changed, only the draft has.
            yield _event(ctx, self.name, "evidence unchanged", actions)
            return

        evidence = gather_evidence(cue)
        block = format_evidence(evidence)
        actions.state_delta["evidence_block"] = block
        actions.state_delta["evidence_phrases"] = len(evidence.phrases)
        actions.state_delta["evidence_neighbours"] = len(evidence.neighbours)
        actions.state_delta["grounded"] = not evidence.is_empty
        actions.state_delta["retry_note"] = ""
        yield _event(
            ctx, self.name,
            f"{len(evidence.phrases)} attested phrases, {len(evidence.neighbours)} neighbours",
            actions,
        )


class RegisterGate(BaseAgent):
    """Accept the draft, or send it back once. The lexicon decides, not the model."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        actions = EventActions()
        state = ctx.session.state
        draft = (state.get("draft") or "").strip().strip('"')
        report = check_register(draft)

        actions.state_delta["register"] = report.summary
        actions.state_delta["not_mexican"] = list(report.not_mexican)
        actions.state_delta["watch"] = list(report.watch)
        actions.state_delta["mexican"] = list(report.mexican)

        if report.ok:
            actions.state_delta["final"] = draft
            actions.escalate = True          # stop the loop: Python decided, not Gemini
            yield _event(ctx, self.name, f"accepted -- {report.summary}", actions)
            return

        # One retry. If the second attempt still fails, the loop ends on max_iterations
        # and the draft is returned with its warning attached rather than silently kept.
        actions.state_delta["final"] = draft
        actions.state_delta["retry_note"] = RETRY_SUFFIX.format(
            markers=", ".join(report.not_mexican), previous=draft
        )
        yield _event(ctx, self.name, f"rejected -- {report.summary}", actions)


def build_localiser(model: str | None = None) -> LoopAgent:
    translator = LlmAgent(
        name="translator",
        model=model or settings().gemini_model,
        description="Writes the Mexican Spanish line from retrieved precedent.",
        instruction=INSTRUCTION + "\n\n{evidence_block}\n{retry_note}",
        generate_content_config=GENERATION_CONFIG,
        output_key="draft",
    )
    return LoopAgent(
        name="localiser",
        description="Retrieve precedent, translate once, and check the register.",
        max_iterations=2,
        sub_agents=[
            SequentialAgent(
                name="localise_pass",
                sub_agents=[
                    EvidenceAgent(name="evidence",
                                  description="Retrieves precedent. Deterministic, no LLM."),
                    translator,
                    RegisterGate(name="register_gate",
                                 description="Checks the register. Deterministic, no LLM."),
                ],
            )
        ],
    )


@dataclass
class Localised:
    """One cue, localised, with everything needed to audit the result."""

    cue: str
    spanish: str = ""
    register: str = ""
    not_mexican: list[str] = field(default_factory=list)
    watch: list[str] = field(default_factory=list)
    mexican: list[str] = field(default_factory=list)
    phrases: int = 0
    neighbours: int = 0
    passes: int = 0
    grounded: bool = True
    evidence_block: str = ""

    @property
    def clean(self) -> bool:
        """No forms Mexicans avoid. Says nothing about whether the line had precedent."""
        return not self.not_mexican

    @property
    def trustworthy(self) -> bool:
        """Clean AND grounded.

        Without the second half a line with no precedent at all reports success: neutral
        Spanish carries no rejectable marker, so the gate passes it, and the result is
        indistinguishable from a grounded one. Across 1,500 cues of a film that silently
        mixes real output with guesses.
        """
        return self.clean and self.grounded


async def localise_async(
    cue: str, *, model: str | None = None, user_id: str = "local",
) -> Localised:
    """Run one cue through the three steps. One Gemini call, two if the gate rejects."""
    runner = InMemoryRunner(agent=build_localiser(model), app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=user_id, state={"cue": cue, "retry_note": ""}
    )

    passes = 0
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=cue)]),
    ):
        if event.author == "translator":
            passes += 1

    final = await runner.session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session.id
    )
    state = final.state if final is not None else {}
    return Localised(
        cue=cue,
        spanish=(state.get("final") or state.get("draft") or "").strip().strip('"'),
        register=state.get("register", ""),
        not_mexican=list(state.get("not_mexican", [])),
        watch=list(state.get("watch", [])),
        mexican=list(state.get("mexican", [])),
        phrases=int(state.get("evidence_phrases", 0)),
        neighbours=int(state.get("evidence_neighbours", 0)),
        passes=passes,
        grounded=bool(state.get("grounded", True)),
        evidence_block=state.get("evidence_block", ""),
    )


def localise(cue: str, *, model: str | None = None) -> Localised:
    """Synchronous wrapper around :func:`localise_async`."""
    import asyncio
    import uuid

    return asyncio.run(localise_async(cue, model=model, user_id=f"cli-{uuid.uuid4().hex[:8]}"))
