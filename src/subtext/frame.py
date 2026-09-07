"""What a scene says about who is speaking to whom.

English `you` is tú, usted and ustedes at once, and nothing in the English line says
which. The Spanish line has to choose, and the corpus shows the same cue taking all three
readings in different films. So the choice cannot be made from the line; it is made from
the scene, and this is the step that makes it.

It is the one step in v2 that genuinely needs a model. Deciding that `Ma'am, do you need
a ride?` puts the next line in `usted` is inference about a social relation, not a lookup
-- there is no lexicon to consult and no morphology to read, because English does not mark
it. Everything else v2 does is retrieval and counting, and stays in Python.

The frame is produced once per scene and reused for every cue in it, which is why the cost
of v2 is about 20% above v1 rather than double. It is a *constraint on retrieval*, not
extra prose in the prompt: precedent that contradicts the frame is ranked down before the
model ever sees it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from .address import Address

FRAME_INSTRUCTION = """\
You are reading consecutive subtitle cues from ONE scene of a film, in English.

Decide how the dialogue would be addressed in MEXICAN Spanish, and return ONLY JSON:

{"address": "tu" | "usted" | "ustedes" | "unknown",
 "confidence": "high" | "low",
 "listeners": <integer, how many people are being addressed>,
 "why": "<at most 12 words, naming the cue that decided it>"}

How to choose:
- "ustedes" — the speaker addresses two or more people. Look for plural addressees named
  or implied ("you guys", "all of you", a group answering).
- "usted" — a formal or distant relation between adults: strangers, a customer and a
  member of staff, an employee to a superior, address by title (sir, ma'am, officer,
  doctor, Mr/Mrs), or marked deference.
- "tu" — familiarity: friends, family, partners, children, colleagues of equal standing,
  insults and intimacy. This is the default between people who know each other.
- "unknown" — the scene genuinely does not say. Use it. A wrong frame is worse than none,
  because it will be used to rank evidence.

Judge the whole scene, not one line. Mexican usage is not Spain's: `usted` is common with
strangers and elders, but two young people meeting are still `tú`.
"""


@dataclass(frozen=True)
class Frame:
    """The discourse frame of one scene."""

    address: Address = Address.UNMARKED
    confident: bool = False
    listeners: int = 1
    why: str = ""

    @property
    def constrains(self) -> bool:
        """Whether this frame is worth ranking evidence with.

        A low-confidence guess is not. Ranking precedent by a frame the model was unsure
        of would take the one decision v2 exists to get right and make it noisier than
        v1's frequency ordering, which is at least unbiased.
        """
        return self.address is not Address.UNMARKED and self.confident


_FORMS = {"tu": Address.TU, "tú": Address.TU,
          "usted": Address.USTED, "ustedes": Address.USTEDES}


def parse_frame(reply: str) -> Frame:
    """Read the model's JSON, or return an empty frame. Never raise on bad output."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return Frame()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Frame()
    if not isinstance(data, dict):
        return Frame()
    address = _FORMS.get(str(data.get("address", "")).strip().lower(), Address.UNMARKED)
    listeners = data.get("listeners", 1)
    return Frame(
        address=address,
        confident=str(data.get("confidence", "")).strip().lower() == "high",
        listeners=int(listeners) if isinstance(listeners, (int, float)) else 1,
        why=str(data.get("why", ""))[:120],
    )


def read_frame(cues: Sequence[str], *, ask=None) -> Frame:
    """Infer the frame for one scene. `ask` is the model call, injected by the caller.

    With no `ask` the frame is empty rather than assumed, and v2 falls back to v1's
    ordering for that scene. That is the honest degradation: without a model there is
    nothing in an English scene that says tú from usted.
    """
    if ask is None or not cues:
        return Frame()
    numbered = "\n".join(f"- {c}" for c in cues)
    return parse_frame(ask(f"{FRAME_INSTRUCTION}\nSCENE:\n{numbered}\n"))
