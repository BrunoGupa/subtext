"""Cut a subtitle file into scenes.

Everything v2 does downstream is per scene, so this has to happen first, and it is the
step the earlier design missed: a plan that says "infer the frame once per scene" assumes
a partition that nothing had computed. An `.srt` is a flat list of cues.

Two signals, and the cheap one goes first:

* **The gap between cues.** `srt.py` already parses `start_ms`/`end_ms`. Dialogue inside a
  scene runs close together; a cut leaves silence. This costs nothing and settles most
  boundaries, so the model is asked only about the ones it cannot settle.
* **The model**, for the rest, over a sliding window with overlap.

The corpus has no timestamps at all -- `mx_corpus` is `pair_id, blk, doc_id, en, es` -- so
when scenes are needed on that side the gap signal is simply absent and every boundary is
a model decision. That asymmetry is real and is not papered over: `segment` takes optional
timings and behaves differently without them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

#: Silence longer than this between two cues is a scene change often enough to act on
#: without asking. Subtitle convention puts continuous dialogue well inside it.
GAP_MS = 4_000

#: Below this, the two cues are certainly the same scene and the model is not consulted.
SAME_SCENE_MS = 1_500

#: How many cues the model sees at once. Enough to judge continuity, small enough that a
#: single wrong answer cannot re-cut the whole file.
WINDOW = 12


@dataclass(frozen=True)
class Scene:
    """A contiguous run of cues taken to share one situation."""

    start: int              # index of the first cue, into the list handed to `segment`
    cues: tuple[str, ...]
    #: True when the model, not the clock, decided this scene began.
    inferred: bool = False

    @property
    def end(self) -> int:
        return self.start + len(self.cues)


SEGMENT_INSTRUCTION = """\
You are given consecutive subtitle cues from one film, numbered. Some of them belong to
the same scene; elsewhere the film cuts to a different place, time or set of characters.

Return ONLY a JSON array of the numbers at which a NEW scene begins. No prose.
The first cue shown is not a new scene unless the numbering starts there.

Judge by what the dialogue shows: a change of addressee, of topic, of location, a jump in
time, a new speaker addressing someone not previously present. Continuous back-and-forth
between the same people is ONE scene however long it runs. When in doubt, do not cut:
a missed cut costs less than a scene split down the middle.

Example output: [4, 9]
"""


def _gap_boundaries(timings: Sequence[tuple[int, int]]) -> tuple[set[int], set[int]]:
    """Split the boundaries into ones the clock settles and ones it does not.

    Returns `(certain_cuts, undecided)` as indices of the cue that would *begin* a scene.
    """
    certain: set[int] = set()
    undecided: set[int] = set()
    for i in range(1, len(timings)):
        gap = timings[i][0] - timings[i - 1][1]
        if gap >= GAP_MS:
            certain.add(i)
        elif gap > SAME_SCENE_MS:
            undecided.add(i)
    return certain, undecided


def _parse_cuts(reply: str, lo: int, hi: int) -> set[int]:
    """Read the model's answer defensively: it is a list of integers or it is nothing."""
    match = re.search(r"\[[^\]]*\]", reply)
    if not match:
        return set()
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    return {int(v) for v in values if isinstance(v, (int, float)) and lo < int(v) < hi}


def segment(
    cues: Sequence[str],
    timings: Sequence[tuple[int, int]] | None = None,
    *,
    ask=None,
) -> list[Scene]:
    """Partition `cues` into scenes.

    `ask` is a callable taking a prompt and returning the model's text, injected so this is
    testable and so the caller owns the model configuration. Passing `ask=None` runs on the
    clock alone, which is the correct fallback when there is no key: it under-segments
    rather than guessing, and an over-long scene degrades v2 towards v1 instead of
    inventing a frame from unrelated lines.
    """
    if not cues:
        return []

    cuts: set[int] = set()
    inferred: set[int] = set()
    if timings:
        certain, undecided = _gap_boundaries(timings)
        cuts |= certain
    else:
        undecided = set(range(1, len(cues)))

    if ask is not None and undecided:
        for start in range(0, len(cues), WINDOW - 2):
            window = cues[start:start + WINDOW]
            if len(window) < 2:
                break
            if not any(start < i < start + len(window) for i in undecided):
                continue
            numbered = "\n".join(f"{start + n}. {c}" for n, c in enumerate(window))
            found = _parse_cuts(ask(f"{SEGMENT_INSTRUCTION}\n{numbered}\n"),
                                start, start + len(window))
            new = found & undecided
            cuts |= new
            inferred |= new

    scenes: list[Scene] = []
    bounds = sorted(cuts | {0, len(cues)})
    for a, b in zip(bounds, bounds[1:]):
        if b > a:
            scenes.append(Scene(start=a, cues=tuple(cues[a:b]), inferred=a in inferred))
    return scenes
