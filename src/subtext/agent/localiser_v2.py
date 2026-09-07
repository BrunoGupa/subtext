"""v2: the same localisation, with the scene in front of it.

v1 (`localiser.py`) translates a cue with no idea what surrounds it. That is enough for
word choice — `auto` or `coche` is a question about Mexican usage, and the corpus answers
it — but it cannot answer the question English does not ask:

    Get in the car.  ->  Súbete al coche.   (tú)
                     ->  Súbase.            (usted)
                     ->  - Entren al carro. (ustedes)

All three are in this corpus, for the same English line, and only the scene says which is
right. `Súbase.` is there because three lines earlier someone said *"Ma'am, do you need a
ride?"*. v1 never sees that line, so it picks by how many translators happened to agree,
which on this cue means it picks `tú` and is wrong a third of the time by construction.

    v1:  cue ──────────────────────────► evidence ─► translate ─► gate
    v2:  file ─► SCENES ─► FRAME ─► cue ─► evidence ─► translate ─► gate
                  (LLM)    (LLM)          (reranked)     (LLM)     (Python,
                  once     once per                                 + form
                  per file  scene                                   check)

Two model calls are added and neither is per cue: segmentation runs once over the file,
the frame once per scene. Over ~1,500 cues that is about 20% more calls, not double.

Everything else stays Python, and deliberately. The tagger that reads tú/usted/ustedes off
a Spanish precedent is morphology, the gate is counting, and the reranking is a sort. Only
the frame is a judgement, because only the frame is asking a question the text does not
already answer. That is the project's own rule, and the corpus supports it: the register
gate loosened for one film would be a model deciding what Mexicans write, which is what
the corpus is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncGenerator, Sequence

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.genai import types

from ..address import Address, read_address
from ..config import settings
from ..frame import Frame, read_frame
from ..localise import (
    INSTRUCTION,
    RETRY_SUFFIX,
    check_register,
    format_evidence,
    gather_evidence,
)
from ..scenes import Scene, segment
from .agent import APP_NAME
from .localiser import GENERATION_CONFIG, Localised, _event


def _asker(model: str | None = None):
    """A plain one-shot call to Gemini, for the two steps that only need an answer.

    Segmentation and frame inference take a prompt and return text. Wrapping either in an
    `LlmAgent` would buy a tool loop neither uses and a session neither needs, so they get
    the raw client with the same generation config the translator runs under -- including
    `thinking_budget=0`, for the same reason: on this task reasoning walks away from the
    evidence.
    """
    from google import genai

    client = genai.Client()
    name = model or settings().gemini_model

    def ask(prompt: str) -> str:
        response = client.models.generate_content(
            model=name, contents=prompt, config=GENERATION_CONFIG
        )
        return response.text or ""

    return ask


@dataclass
class ScenedCue:
    """One cue with the scene it belongs to."""

    index: int
    cue: str
    scene: int
    frame: Frame


def plan_scenes(
    cues: Sequence[str],
    timings: Sequence[tuple[int, int]] | None = None,
    *,
    ask=None,
) -> list[ScenedCue]:
    """Steps 0 and 1: cut the file into scenes, then read each scene's frame.

    Runs before any cue is translated, because the frame is shared by every cue in the
    scene and asking per cue would pay for it 1,500 times instead of 150.
    """
    scenes: list[Scene] = segment(cues, timings, ask=ask)
    planned: list[ScenedCue] = []
    for number, scene in enumerate(scenes):
        frame = read_frame(scene.cues, ask=ask)
        for offset, cue in enumerate(scene.cues):
            planned.append(ScenedCue(index=scene.start + offset, cue=cue,
                                     scene=number, frame=frame))
    return planned


class FramedEvidenceAgent(BaseAgent):
    """Retrieve precedent, ranked by the scene's form of address. No model, no decisions."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        actions = EventActions()
        state = ctx.session.state
        cue = state.get("cue", "")

        if state.get("evidence_block"):
            yield _event(ctx, self.name, "evidence unchanged", actions)
            return

        frame = state.get("frame") or Frame()
        evidence = gather_evidence(cue, frame=frame)
        actions.state_delta["evidence_block"] = format_evidence(evidence)
        actions.state_delta["evidence_phrases"] = len(evidence.phrases)
        actions.state_delta["evidence_neighbours"] = len(evidence.neighbours)
        actions.state_delta["grounded"] = not evidence.is_empty
        actions.state_delta["retry_note"] = ""
        note = f"frame {frame.address.value}" if frame.constrains else "no frame"
        yield _event(ctx, self.name,
                     f"{len(evidence.phrases)} phrases, "
                     f"{len(evidence.neighbours)} neighbours, {note}", actions)


class FramedRegisterGate(BaseAgent):
    """v1's register check, plus: does the output address the listener as the scene does?

    The second check is what makes the frame binding rather than advisory. A frame that
    only reorders evidence can still be ignored by the model; a frame that can send the
    line back cannot. It fires only on a confident frame and only when the output states a
    form -- an unmarked line is not wrong, since Spanish drops the pronoun in about 79% of
    plural-you lines and forcing one would be its own error.
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        actions = EventActions()
        state = ctx.session.state
        draft = (state.get("draft") or "").strip().strip('"')
        report = check_register(draft)
        frame: Frame = state.get("frame") or Frame()
        reading = read_address(draft)

        mismatch = (
            frame.constrains
            and reading.marked
            and reading.form is not frame.address
            and reading.form is not Address.VOSOTROS   # already caught as NOT_MEXICAN
        )

        actions.state_delta["register"] = report.summary
        actions.state_delta["not_mexican"] = list(report.not_mexican)
        actions.state_delta["watch"] = list(report.watch)
        actions.state_delta["mexican"] = list(report.mexican)
        actions.state_delta["address"] = reading.form.value
        actions.state_delta["address_matches_scene"] = not mismatch

        if report.ok and not mismatch:
            actions.state_delta["final"] = draft
            actions.escalate = True
            yield _event(ctx, self.name, f"accepted -- {report.summary}", actions)
            return

        actions.state_delta["final"] = draft
        if mismatch:
            actions.state_delta["retry_note"] = (
                f"\n\nYour previous attempt addressed the listener as "
                f"**{reading.form.value}**, but this scene is **{frame.address.value}** "
                f"({frame.why}). Rewrite the line using {frame.address.value}, keeping the "
                f"same meaning, wording and length. Change nothing else.\n"
                f"Previous attempt: {draft}\n"
            )
            reason = f"address {reading.form.value} != scene {frame.address.value}"
        else:
            actions.state_delta["retry_note"] = RETRY_SUFFIX.format(
                markers=", ".join(report.not_mexican), previous=draft
            )
            reason = report.summary
        yield _event(ctx, self.name, f"rejected -- {reason}", actions)


def build_localiser_v2(model: str | None = None) -> LoopAgent:
    translator = LlmAgent(
        name="translator",
        model=model or settings().gemini_model,
        description="Writes the Mexican Spanish line from scene-ranked precedent.",
        instruction=INSTRUCTION + "\n\n{evidence_block}\n{retry_note}",
        generate_content_config=GENERATION_CONFIG,
        output_key="draft",
    )
    return LoopAgent(
        name="localiser_v2",
        description="Scene frame, ranked precedent, one translation, two checks.",
        max_iterations=2,
        sub_agents=[
            SequentialAgent(
                name="localise_pass_v2",
                sub_agents=[
                    FramedEvidenceAgent(
                        name="framed_evidence",
                        description="Retrieves precedent ranked by scene. No LLM."),
                    translator,
                    FramedRegisterGate(
                        name="framed_gate",
                        description="Checks register and form of address. No LLM."),
                ],
            )
        ],
    )


@dataclass
class LocalisedV2(Localised):
    """A v1 result plus what the scene contributed."""

    scene: int = 0
    scene_address: str = Address.UNMARKED.value
    scene_why: str = ""
    address: str = Address.UNMARKED.value
    address_matches_scene: bool = True


async def localise_framed_async(
    cue: str, frame: Frame | None = None, *,
    scene: int = 0, model: str | None = None, user_id: str = "local",
) -> LocalisedV2:
    """Run one cue through v2. The frame comes from `plan_scenes`, computed once."""
    frame = frame or Frame()
    runner = InMemoryRunner(agent=build_localiser_v2(model), app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=user_id,
        state={"cue": cue, "retry_note": "", "frame": frame},
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
    return LocalisedV2(
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
        scene=scene,
        scene_address=frame.address.value,
        scene_why=frame.why,
        address=str(state.get("address", Address.UNMARKED.value)),
        address_matches_scene=bool(state.get("address_matches_scene", True)),
    )


def localise_file(
    cues: Sequence[str],
    timings: Sequence[tuple[int, int]] | None = None,
    *,
    model: str | None = None,
) -> list[LocalisedV2]:
    """The whole v2 pipeline over a file: segment, frame, then translate cue by cue."""
    import asyncio
    import uuid

    ask = _asker(model)
    planned = plan_scenes(cues, timings, ask=ask)

    async def run() -> list[LocalisedV2]:
        out = []
        for item in planned:
            out.append(await localise_framed_async(
                item.cue, item.frame, scene=item.scene, model=model,
                user_id=f"v2-{uuid.uuid4().hex[:8]}"))
        return out

    return asyncio.run(run())
